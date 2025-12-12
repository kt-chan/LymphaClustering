import numpy as np
import os
import json
import uuid
import time
import threading
from datetime import datetime
from glob import glob
from typing import List, Dict, Any, Optional

# --- API & Server ---
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi_standalone_docs import StandaloneDocs
from pydantic import BaseModel
from starlette.responses import JSONResponse
from scipy import ndimage

# --- ML & Data (PyTorch / Hugging Face) ---
import torch
from transformers import AutoImageProcessor, ResNetModel
from PIL import Image
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

# --- Custom modules ---
from app_logger import app_logger
from db_manager import VectorDBManager, VECTOR_DIM

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = os.getenv("DATA_DIR", r"D:/data/ruijin/Data/crops")

# NOTE: For HuggingFace offline mode, this path should be a FOLDER containing 
# config.json, pytorch_model.bin, and preprocessor_config.json
RESNET50_LOCAL_PATH = os.getenv(
    "RESNET50_LOCAL_PATH", r"D:/models/resnet-50"
)

MASK_OUTPUT_DIR = "masks"
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"
DEFAULT_DB_PATH = os.getenv("DEFAULT_DB_PATH", "lancedb_histology")
DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "histology_specimens")
DEFAULT_THRESHOLD = 0.36

# Updated to 5% increment as requested
THRESHOLD_INCREMENT = 1.02
THRESHOLD_INCREMENT_DEPTH = 3

# Target size is handled by the Processor, but useful to keep for consistency if needed
IMAGE_TARGET_SIZE = (224, 224) 
BATCH_SIZE = 32

# Clustering Settings
BUFFER_FLUSH_INTERVAL_SECONDS = 60
BUFFER_BATCH_LIMIT = 2000

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Models & Request Objects
# =============================================================================


class TuningRequest(BaseModel):
    path: str = ""
    ground_truth_csv: str = "crops_label.csv"
    early_stop_patience: int = 10
    early_stop_stepsize: Optional[float] = 0.05


class DirectoryRequest(BaseModel):
    path: str = ""
    threshold: Optional[float] = DEFAULT_THRESHOLD
    recursive_cluster: Optional[bool] = True


class SimilarityRequest(BaseModel):
    slide_id: str
    limit: int = 5


class MaintenanceRequest(BaseModel):
    force: bool = False


# Initialize Global Manager
db_manager = VectorDBManager(
    DEFAULT_DB_PATH, DEFAULT_TABLE_NAME, flush_interval=BUFFER_FLUSH_INTERVAL_SECONDS, batch_limit=BUFFER_BATCH_LIMIT
)

app = FastAPI(title="Histology Clustering API (PyTorch)", version="1.2.0")
StandaloneDocs(app=app)


@app.on_event("startup")
def startup_tasks():
    """Initialize application on startup."""
    # 1. Load Model (PyTorch / Hugging Face)
    app_logger.info(f"Loading ResNet50 model from {RESNET50_LOCAL_PATH} on {DEVICE}...")
    
    try:
        # Load Processor and Model from local directory
        processor = AutoImageProcessor.from_pretrained(RESNET50_LOCAL_PATH, local_files_only=True)
        # We use ResNetModel (Base) to get feature vectors (avg pooling), not class logits
        model = ResNetModel.from_pretrained(RESNET50_LOCAL_PATH, local_files_only=True)
        
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        
        app.state.processor = processor
        app.state.model = model
        app_logger.info("Model loaded successfully.")
    except Exception as e:
        app_logger.error(f"Failed to load model: {e}")
        raise e

    # 2. Start Maintenance Scheduler (Daily Compaction)
    def daily_optimize():
        while True:
            time.sleep(86400)  # 24 hours
            try:
                db_manager.run_maintenance()
            except Exception as e:
                app_logger.error(f"Maintenance failed: {e}")

    t = threading.Thread(target=daily_optimize, daemon=True)
    t.start()


# =============================================================================
# Core Logic Functions
# =============================================================================


def _smart_extract_features(model: ResNetModel, image_paths: List[str], threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Extract features from images with caching via VectorDB."""
    path_map = {os.path.basename(p): p for p in image_paths}
    all_slide_ids = list(path_map.keys())

    # 1. Check Cache (Hybrid: Disk + Buffer)
    cached_df = db_manager.get_existing_vectors(all_slide_ids)

    results = []
    cached_ids = set()

    if not cached_df.empty:
        cached_ids = set(cached_df["slide_id"].tolist())
        for _, row in cached_df.iterrows():
            results.append(
                {
                    "slide_id": row["slide_id"],
                    "wsi_id": row["wsi_id"],
                    "vector": row["vector"],
                    "path": path_map.get(row["slide_id"], row["path"]),
                    "timestamp": row["timestamp"],
                }
            )

    # Debug mode to bypass cache
    if DEBUG_MODE:
        results = []
        cached_ids = set()

    # 2. Compute Missing
    missing_ids = [sid for sid in all_slide_ids if sid not in cached_ids]

    app_logger.info(f"Cache hit rate: {len(cached_ids)} / {len(image_paths)}")

    if missing_ids:
        batch_images = [] # This will hold PIL Images
        batch_meta = []
        new_records_for_buffer = []
        processor = app.state.processor

        app_logger.info(f"Performing vectorization for {len(missing_ids)} images.")

        for i, sid in enumerate(missing_ids):
            p = path_map[sid]
            try:
                # Use the preprocessing function (returns PIL Image)
                processed_img = _pre_process_image(p)
                batch_images.append(processed_img)
                batch_meta.append(
                    {
                        "slide_id": sid,
                        "wsi_id": os.path.basename(os.path.dirname(p)),
                        "path": p,
                        "timestamp": datetime.now(),
                    }
                )
            except Exception as e:
                app_logger.warning(f"Bad image {p}: {e}")

            # Process Batch
            if len(batch_images) == BATCH_SIZE or (i == len(missing_ids) - 1 and batch_images):
                try:
                    # HF Processor handles resizing and normalization
                    inputs = processor(images=batch_images, return_tensors="pt")
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    # Get pooled output (Batch, 2048, 1, 1) -> Flatten to (Batch, 2048)
                    # ResNetModel output usually has 'pooler_output' or 'last_hidden_state'
                    # We want pooler_output for global average pooling equivalent
                    embeddings = outputs.pooler_output.squeeze()
                    
                    # Handle case where batch size is 1 (squeeze removes too many dims)
                    if len(embeddings.shape) == 1:
                        embeddings = embeddings.unsqueeze(0)

                    preds = embeddings.cpu().numpy()

                    for j, meta in enumerate(batch_meta):
                        meta["vector"] = preds[j].tolist()
                        meta["cluster_id"] = -1
                        meta["threshold"] = threshold
                        results.append(meta)
                        new_records_for_buffer.append(meta)
                
                except Exception as e:
                    app_logger.error(f"Inference error on batch: {e}")
                
                # Clear batch
                batch_images = []
                batch_meta = []

        # 3. Add to Buffer
        if new_records_for_buffer:
            db_manager.buffer.add(new_records_for_buffer)

    return pd.DataFrame(results)


def _pre_process_image(img_path: str, greyScale: bool = False) -> Image.Image:
    """
    Preprocess an image by loading it using PIL.
    Returns a PIL Image object (processor handles array conversion).
    """
    try:
        # Load image with PIL
        img = Image.open(img_path)
        
        # Handle grayscale / color conversion
        if greyScale or img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img

    except Exception as e:
        raise ValueError(f"Failed to preprocess image {img_path}: {e}")


def _reassign_orphans_dynamic(wsi_df: pd.DataFrame, initial_threshold: float = 0.5) -> pd.DataFrame:
    """
    Dynamically identifies orphan clusters and merges them.
    Includes recursive re-clustering of orphans with increasing thresholds (5% incr, max 3 times).
    Validates result using Silhouette Score; falls back if score decreases.
    """
    if wsi_df.empty:
        return wsi_df

    # Helper: Calculate Silhouette Score
    def _calculate_score(df: pd.DataFrame) -> float:
        """Calculates Silhouette Score for current clusters."""
        labels = df["cluster_id"].values
        unique_labels = set(labels)

        # Silhouette Score requires at least 2 clusters and more samples than clusters
        if len(unique_labels) < 2 or len(df) <= len(unique_labels):
            return -1.0

        try:
            vectors = np.vstack(df["vector"].values)
            # Normalize to match the metric used in AgglomerativeClustering (Euclidean on normalized data)
            vectors_norm = normalize(vectors, axis=1, norm="l2")
            return silhouette_score(vectors_norm, labels, metric="euclidean")
        except Exception:
            return -1.0

    # --- Step 1: Baseline Score ---
    # Initialize final_threshold for all points if missing
    if "threshold" not in wsi_df.columns:
        wsi_df["threshold"] = initial_threshold

    original_score = _calculate_score(wsi_df)
    app_logger.debug(f"Baseline Silhouette Score: {original_score:.4f}")

    # --- Step 2: Recursive Logic Definition ---

    def _recursive_orphan_handler(sub_df: pd.DataFrame, current_thresh: float, depth: int, max_depth: int):
        """Helper to recursively re-cluster orphans."""
        if sub_df.empty:
            return sub_df, []

        sub_df = sub_df.copy()

        # Extract vectors for current batch of orphans
        vectors = np.vstack(sub_df["vector"].values)

        # If we have too few samples to cluster, they remain orphans
        if len(vectors) < 2:
            sub_df["threshold"] = current_thresh
            return sub_df, sub_df.index.tolist()

        clustering_input = normalize(vectors, axis=1, norm="l2")

        # Attempt clustering with increased threshold
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=current_thresh, metric="euclidean", linkage="average"
        )
        labels = model.fit_predict(clustering_input)

        sub_df["temp_cluster_id"] = labels

        # Check size of new clusters
        counts = sub_df["temp_cluster_id"].value_counts()

        # Identify which points are still orphans (size 1)
        still_orphans_mask = sub_df["temp_cluster_id"].map(counts) == 1

        resolved_df = sub_df[~still_orphans_mask].copy()
        remaining_orphans_df = sub_df[still_orphans_mask].copy()

        # Update final_threshold for resolved points
        if not resolved_df.empty:
            resolved_df["threshold"] = current_thresh

        # Recurse if we haven't hit max depth and still have orphans
        if not remaining_orphans_df.empty and depth < max_depth:
            # Increase threshold by 5%
            new_thresh = current_thresh * THRESHOLD_INCREMENT
            processed_orphans, final_orphan_indices = _recursive_orphan_handler(
                remaining_orphans_df, new_thresh, depth + 1, max_depth
            )
            return pd.concat([resolved_df, processed_orphans]), final_orphan_indices
        else:
            # End of recursion
            if not remaining_orphans_df.empty:
                remaining_orphans_df["threshold"] = current_thresh

            return pd.concat([resolved_df, remaining_orphans_df]), remaining_orphans_df.index.tolist()

    # --- Step 3: Identify Orphans & Run Recursion ---

    counts = wsi_df["cluster_id"].value_counts()
    if counts.empty:
        return wsi_df

    # Determine Typical Size (Mode)
    typical_size = counts.max()

    # Identify Initial Orphans (<= half of typical size)
    orphan_threshold_size = typical_size / 2.0
    orphan_cluster_ids = counts[counts <= orphan_threshold_size].index.tolist()

    if not orphan_cluster_ids:
        return wsi_df

    # Separate healthy clusters from orphans
    orphans_df = wsi_df[wsi_df["cluster_id"].isin(orphan_cluster_ids)].copy()
    healthy_df = wsi_df[~wsi_df["cluster_id"].isin(orphan_cluster_ids)].copy()

    # Run Recursive Re-clustering
    re_clustered_orphans, final_orphan_indices = _recursive_orphan_handler(
        orphans_df, current_thresh=initial_threshold, depth=0, max_depth=THRESHOLD_INCREMENT_DEPTH
    )

    # --- Step 4: Integrate & Reassign Stubborn Orphans ---

    max_id = healthy_df["cluster_id"].max() if not healthy_df.empty else -1

    # Remap IDs for the re-clustered group to avoid collision
    if "temp_cluster_id" in re_clustered_orphans.columns:
        new_ids = re_clustered_orphans["temp_cluster_id"].unique()
        id_map = {old_id: max_id + 1 + i for i, old_id in enumerate(new_ids)}
        re_clustered_orphans["cluster_id"] = re_clustered_orphans["temp_cluster_id"].map(id_map)
        re_clustered_orphans.drop(columns=["temp_cluster_id"], inplace=True)

    # Create the candidate dataframe
    proposed_df = pd.concat([healthy_df, re_clustered_orphans])

    # Recalculate centroids for valid clusters (exclude stubborn orphans)
    centroids = {}
    valid_unique_clusters = proposed_df["cluster_id"].unique()

    for cid in valid_unique_clusters:
        cluster_indices = proposed_df[proposed_df["cluster_id"] == cid].index
        # If this cluster is purely made of stubborn orphans (failed recursion), skip centroid calc
        if set(cluster_indices).issubset(set(final_orphan_indices)):
            continue

        cluster_vectors = np.vstack(proposed_df[proposed_df["cluster_id"] == cid]["vector"].values)
        centroids[cid] = np.mean(cluster_vectors, axis=0)

    # Assign remaining stubborn orphans to nearest valid centroid
    if centroids:
        for idx in final_orphan_indices:
            if idx in proposed_df.index:
                current_vector = proposed_df.loc[idx, "vector"]

                min_dist = float("inf")
                best_cid = proposed_df.loc[idx, "cluster_id"]  # Default to self

                for target_cid, target_centroid in centroids.items():
                    dist = np.linalg.norm(current_vector - target_centroid)
                    if dist < min_dist:
                        min_dist = dist
                        best_cid = target_cid

                proposed_df.at[idx, "cluster_id"] = best_cid

    # --- Step 5: Validation (Fallback) ---

    new_score = _calculate_score(proposed_df)
    app_logger.debug(f"New Silhouette Score: {new_score:.4f}")

    if new_score > original_score:
        app_logger.debug(f"Reassignment successful. Score improved: {original_score:.4f} -> {new_score:.4f}")
        return proposed_df
    else:
        app_logger.debug(
            f"Reassignment fallback. Score did not improve ({new_score:.4f} vs {original_score:.4f}). Keeping original."
        )
        return wsi_df


def _run_clustering(df: pd.DataFrame, threshold: float, recursive: bool = True) -> pd.DataFrame:
    """Perform hierarchical clustering on the feature vectors."""
    final_dfs = []
    if df.empty:
        return df

    app_logger.info(f"Performing clustering for {len(df)} records ... ")

    grouped = df.groupby("wsi_id")
    for wsi_id, group in grouped:
        wsi_df = group.copy()
        if len(wsi_df) < 2:
            wsi_df["cluster_id"] = 0
            wsi_df["threshold"] = threshold
            final_dfs.append(wsi_df)
            continue

        vectors = np.vstack(wsi_df["vector"].values)
        clustering_input = normalize(vectors, axis=1, norm="l2")
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="average"
        )
        wsi_df["cluster_id"] = model.fit_predict(clustering_input)

        # Pass the initial threshold to the dynamic reassigner
        if recursive:
            final_wsi_df = _reassign_orphans_dynamic(wsi_df, initial_threshold=threshold)
        else:
            wsi_df["threshold"] = threshold
            final_wsi_df = wsi_df

        final_dfs.append(final_wsi_df)

    return pd.concat(final_dfs, ignore_index=True)


# =============================================================================
# API Endpoints
# =============================================================================


@app.post("/v1/cluster/from_directory", response_class=JSONResponse)
async def cluster_from_directory(request: DirectoryRequest = Body(...)):
    """Cluster images from a directory based on visual similarity."""
    try:
        search_root = os.path.join(DATA_DIR, request.path)

        # 1. Glob Images
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.tif"]:
            image_paths.extend(glob(os.path.join(search_root, "**", ext), recursive=True))

        if not image_paths:
            raise HTTPException(404, "No images found.")

        app_logger.info(f"cluster_from_directory for {len(image_paths)} images.")
        threshold = request.threshold or DEFAULT_THRESHOLD
        recursive_cluster = request.recursive_cluster

        # 2. Smart Extract
        df_data = _smart_extract_features(app.state.model, image_paths, threshold)

        # 3. Clustering
        clustered_df = _run_clustering(df_data, threshold, recursive_cluster)

        # 4. Async Save
        db_manager.buffer.add(clustered_df.to_dict(orient="records"))

        # 5. Response
        results = []
        for wsi_id, w_group in clustered_df.groupby("wsi_id"):
            clusters = []
            for c_id, c_group in w_group.groupby("cluster_id"):

                # Calculate max threshold used in this cluster
                c_thresh = c_group["threshold"].max() if "threshold" in c_group else 0.0

                clusters.append(
                    {
                        "cluster_id": int(c_id),
                        "count": len(c_group),
                        "threshold": float(c_thresh),
                        "slides": c_group["slide_id"].tolist(),
                    }
                )
            results.append(
                {
                    "wsi_id": wsi_id,
                    "clusters_count": len(clusters),
                    "clusters_details": clusters,
                    "debug": f"{wsi_id},{len(clusters)}",
                }
            )
        output = {"id": str(uuid.uuid4()), "data": results}
        app_logger.info(f"Cluster output: {output}")
        return output

    except Exception as e:
        app_logger.error(f"Cluster Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/tune_parameters")
async def tune_parameters(request: TuningRequest = Body(...)):
    """Tune clustering parameters using ground truth data."""
    try:
        search_root = os.path.join(DATA_DIR, request.path)
        gt_path = os.path.join(search_root, request.ground_truth_csv)
        if not os.path.exists(gt_path):
            raise HTTPException(404, "Ground truth CSV not found.")

        test_thresholds = np.arange(0.01, 1.01, request.early_stop_stepsize).tolist()

        gt_df = pd.read_csv(gt_path)
        gt_df.columns = [c.strip() for c in gt_df.columns]
        ground_truth_map = dict(zip(gt_df["image_name"], gt_df["lymph_node_total_num"]))

        image_paths = []
        for ext in ["*.png"]:
            image_paths.extend(glob(os.path.join(search_root, "**", ext), recursive=True))

        app_logger.info(f"Performing tune_parameters for {len(image_paths)} images")

        relevant_paths = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in ground_truth_map]

        app_logger.info(f"Vectorizing {len(relevant_paths)} images...")
        data_df = _smart_extract_features(app.state.model, relevant_paths)
        app_logger.info(f"Images vectorization done for {len(data_df)} images.")

        # Tuning with Early Stopping
        results = []
        patience = request.early_stop_patience
        best_accuracy = 0
        patience_counter = 0

        for thresh in test_thresholds:
            app_logger.info(f"Testing threshold: {thresh:.3f}")
            temp_df = _run_clustering(data_df, threshold=thresh)
            total_error = 0
            wsi_count = 0
            hit_count = 0

            for wsi_id, group in temp_df.groupby("wsi_id"):
                if wsi_id in ground_truth_map:
                    pred_k = group["cluster_id"].nunique()
                    actual_k = ground_truth_map[wsi_id]

                    # Adjust for single-image cases
                    if len(group) == 1 and pred_k == 1:
                        actual_k = 1

                    abs_error = abs(pred_k - actual_k)
                    if abs_error == 0:
                        hit_count += 1
                    total_error += abs_error
                    wsi_count += 1

            if wsi_count > 0:
                current_accuracy = round(hit_count / wsi_count, 4)
                current_mae = round(total_error / wsi_count, 4)

                results.append(
                    {
                        "threshold": round(thresh, 4),
                        "accuracy": current_accuracy,
                        "mae": current_mae,
                        "hit_count": hit_count,
                        "wsi_count": wsi_count,
                    }
                )

                # Early stopping logic
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    patience_counter = 0
                    app_logger.info(f"New best accuracy: {best_accuracy} at threshold {thresh:.3f}")
                else:
                    patience_counter += 1
                    app_logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    app_logger.info(f"Early stopping triggered at threshold {thresh:.3f}")
                    break
            else:
                app_logger.warning(f"No valid WSI for threshold {thresh:.3f}")

        if not results:
            raise HTTPException(500, "No valid results generated during tuning.")

        # Find best result
        results.sort(key=lambda x: (x["accuracy"], -x["mae"]), reverse=True)
        best_result = results[0]

        return {
            "best_parameter": best_result,
            "all_results": sorted(results, key=lambda x: x["threshold"]),
            "early_stopped": len(results) < len(test_thresholds),
            "thresholds_tested": len(results),
        }

    except Exception as e:
        app_logger.error(f"Tuning Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/find_similar_nodes")
async def find_similar_nodes(request: SimilarityRequest = Body(...)):
    """Find similar nodes based on vector similarity."""
    try:
        # 1. Get source
        source_df = db_manager.get_existing_vectors([request.slide_id])
        if source_df.empty:
            raise HTTPException(404, f"Slide ID {request.slide_id} not found.")

        query_vector = source_df.iloc[0]["vector"]

        # 2. Search
        similar_df = db_manager.search_similarity(query_vector, limit=request.limit + 1, exclude_id=request.slide_id)

        # 3. Format
        matches = []
        for _, row in similar_df.iterrows():
            matches.append(
                {
                    "slide_id": row["slide_id"],
                    "wsi_id": row["wsi_id"],
                    "distance": float(row["_distance"]) if "_distance" in row else 0.0,
                    "path": row["path"],
                }
            )

        return {"query_id": request.slide_id, "matches": matches[: request.limit]}

    except Exception as e:
        app_logger.error(f"Search Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/v1/flush_buffer")
async def flush_buffer():
    """Force flush the write buffer to disk."""
    try:
        db_manager.buffer.flush()
        return {"status": "success", "message": "Buffer flushed to disk"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/v1/maintenance")
async def run_maintenance(request: MaintenanceRequest = Body(...)):
    """Run database maintenance tasks."""
    try:
        app_logger.info(f"Manual maintenance requested with force={request.force}")
        results = db_manager.run_maintenance(force=request.force)
        return {"status": "success", "message": "Database maintenance completed", "results": results}
    except Exception as e:
        app_logger.error(f"Maintenance Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.get("/v1/db_stats")
async def get_database_stats():
    """Get comprehensive database statistics."""
    try:
        stats = db_manager.get_system_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/v1/logs/info")
async def get_logs_info():
    """Get information about log rotation settings."""
    try:
        from app_logger import get_rotation_info

        info = get_rotation_info()
        return {"status": "success", "log_info": info}
    except Exception as e:
        app_logger.error(f"Error getting log info: {e}")
        raise HTTPException(500, str(e))


@app.post("/v1/logs/rotate")
async def rotate_logs():
    """Manually rotate log files."""
    try:
        from app_logger import rotate_logs_now

        success = rotate_logs_now()
        if success:
            return {"status": "success", "message": "Logs rotated successfully"}
        else:
            return {"status": "error", "message": "Failed to rotate logs"}
    except Exception as e:
        app_logger.error(f"Error rotating logs: {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), reload=DEBUG_MODE)