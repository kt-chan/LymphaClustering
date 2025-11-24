import numpy as np
import os
import json
import uuid
import logging
import time
from datetime import datetime
from glob import glob
from typing import List, Dict, Any, Optional

# --- API & Server ---
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from starlette.responses import JSONResponse

# --- ML & Data ---
import lancedb
from lancedb.pydantic import LanceModel, Vector
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# --- Paths & Settings ---
DATA_DIR = os.getenv("DATA_DIR", r"D:/data/ruijin/Data/crops")
RESNET50_LOCAL_PATH = os.getenv(
    "RESNET50_LOCAL_PATH", r"D:/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
DEBUG_MODE = os.getenv("DEBUG", False)
DEFAULT_DB_PATH = os.getenv("DEFAULT_DB_PATH", "lancedb_histology")
DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "histology_specimens")
IMAGE_TARGET_SIZE = (224, 224)
BATCH_SIZE = 32  # Optimization: Batch size for inference
VECTOR_DIM = 2048  # ResNet50 Avg Pooling dimension

# =============================================================================
# Logging
# =============================================================================
logging.getLogger("uvicorn.access").disabled = True
app_logger = logging.getLogger("api.application")
app_logger.setLevel(logging.INFO)
app_logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console & File Handlers
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
app_logger.addHandler(console_handler)
app_logger.addHandler(file_handler)
app_logger.propagate = False

# =============================================================================
# 1. Database Schema & Manager (New Architecture)
# =============================================================================


class LymphNodeSchema(LanceModel):
    """
    Strict Schema for LanceDB.
    Enforces data integrity and allows for fast vector search.
    """

    slide_id: str  # Primary Key (filename)
    wsi_id: str  # Foreign Key equivalent (Patient/Slide ID)
    vector: Vector(VECTOR_DIM)  # 2048-dim embedding
    cluster_id: int = -1  # -1 implies not yet clustered
    timestamp: datetime  # Audit trail
    path: str  # File location


class VectorDBManager:
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)
        self.tbl = self._init_table()

    def _init_table(self):
        if self.table_name not in self.db.table_names():
            app_logger.info(f"Creating new Vector Table: {self.table_name}")
            return self.db.create_table(self.table_name, schema=LymphNodeSchema)
        else:
            return self.db.open_table(self.table_name)

    def get_existing_vectors(self, slide_ids: List[str]) -> pd.DataFrame:
        """
        Performance Cache: Retrieve vectors for slide_ids that already exist.
        """
        if not slide_ids:
            return pd.DataFrame()

        # In SQL: SELECT * FROM table WHERE slide_id IN (...)
        # LanceDB optimization: handle large lists by chunks if necessary
        try:
            # Note: For massive lists (>10k), simpler to filter post-fetch or chunk query
            quoted = [f"'{s}'" for s in slide_ids]
            if len(quoted) > 2000:
                # Fallback for huge batches: Fetch by WSI or fetch all (simplified for this context)
                app_logger.info("Batch too large for IN clause, fetching larger slice...")
                return self.tbl.to_pandas()

            filter_str = f"slide_id IN ({','.join(quoted)})"
            return self.tbl.search().where(filter_str).to_pandas()
        except Exception as e:
            app_logger.warning(f"Cache lookup failed, recomputing: {e}")
            return pd.DataFrame()

    def upsert_data(self, df: pd.DataFrame):
        """
        Operability: Updates existing records (e.g. new cluster IDs)
        and inserts new images without wiping the DB.
        """
        if df.empty:
            return

        app_logger.info(f"Upserting {len(df)} records to LanceDB...")

        # Convert timestamps to proper datetime objects if they are strings
        if "timestamp" in df.columns and df["timestamp"].dtype == "O":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        try:
            # Use the newer LanceDB upsert API if available
            if hasattr(self.tbl, "upsert"):
                self.tbl.upsert(df)
            else:
                # Fallback to delete + insert pattern
                existing_ids = set(self.tbl.to_pandas()["slide_id"].tolist())
                update_ids = set(df["slide_id"].tolist()) & existing_ids

                # Delete existing records that need updating
                if update_ids:
                    id_list = ", ".join([f"'{sid}'" for sid in update_ids])
                    self.tbl.delete(f"slide_id IN ({id_list})")

                # Insert all records (both new and updated)
                self.tbl.add(df)

        except Exception as e:
            app_logger.error(f"Upsert failed: {e}")
            # Final fallback
            self.tbl.add(df)

        def search_similarity(self, query_vector: np.array, limit: int = 5, exclude_id: str = None):
            """
            Global Search: Find similar nodes across ALL patients.
            """
            search_builder = self.tbl.search(query_vector).limit(limit)
            if exclude_id:
                search_builder = search_builder.where(f"slide_id != '{exclude_id}'")
            return search_builder.to_pandas()


# Initialize DB Manager Globally
db_manager = VectorDBManager(DEFAULT_DB_PATH, DEFAULT_TABLE_NAME)

# =============================================================================
# 2. Models & Request Objects
# =============================================================================


class TuningRequest(BaseModel):
    path: str = ""
    ground_truth_csv: str = "crops_label.csv"


class DirectoryRequest(BaseModel):
    path: str = ""
    threshold: Optional[float] = 0.35


class SimilarityRequest(BaseModel):
    slide_id: str
    limit: int = 5


app = FastAPI(title="Histology Clustering API (Vector DB Enhanced)", version="2.0.0")


@app.on_event("startup")
def load_model():
    app_logger.info("Loading ResNet50 model...")
    # Initialize without top layer for feature extraction
    base_model = ResNet50(
        weights=RESNET50_LOCAL_PATH, include_top=False, input_shape=(*IMAGE_TARGET_SIZE, 3), pooling="avg"
    )
    app.state.feature_extractor = base_model
    app_logger.info("Model loaded successfully.")


# =============================================================================
# 3. Core Logic (Smart Extraction & Clustering)
# =============================================================================


def _smart_extract_features(model: Model, image_paths: List[str]) -> pd.DataFrame:
    """
    Hybrid Logic:
    1. Check LanceDB cache for vectors.
    2. Compute missing vectors using GPU batch processing.
    3. Save new vectors immediately.
    """
    path_map = {os.path.basename(p): p for p in image_paths}
    all_slide_ids = list(path_map.keys())

    # 1. Check Cache
    app_logger.info(f"Checking cache for {len(all_slide_ids)} images...")
    cached_df = db_manager.get_existing_vectors(all_slide_ids)

    results = []
    cached_ids = set()

    if not cached_df.empty:
        cached_ids = set(cached_df["slide_id"].tolist())
        # Convert cached rows to list of dicts
        for _, row in cached_df.iterrows():
            results.append(
                {
                    "slide_id": row["slide_id"],
                    "wsi_id": row["wsi_id"],
                    "vector": row["vector"],
                    "path": path_map.get(row["slide_id"], row["path"]),  # Update path if moved
                    "timestamp": row["timestamp"],
                }
            )

    app_logger.info(f"Cache hit: {len(cached_ids)} | Missing: {len(all_slide_ids) - len(cached_ids)}")

    # 2. Compute Missing
    missing_ids = [sid for sid in all_slide_ids if sid not in cached_ids]

    if missing_ids:
        # Batch Processing Loop
        batch_images = []
        batch_meta = []

        for i, sid in enumerate(missing_ids):
            p = path_map[sid]
            try:
                img = image.load_img(p, target_size=IMAGE_TARGET_SIZE)
                x = image.img_to_array(img)
                x = preprocess_input(x)  # ResNet preprocessing
                batch_images.append(x)
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

            # Process batch if full or last item
            if len(batch_images) == BATCH_SIZE or (i == len(missing_ids) - 1 and batch_images):
                # Inference
                batch_arr = np.array(batch_images)
                preds = model.predict(batch_arr, verbose=0)

                new_records = []
                for j, meta in enumerate(batch_meta):
                    meta["vector"] = preds[j].tolist()
                    meta["cluster_id"] = -1  # Default
                    results.append(meta)
                    new_records.append(meta)

                # Immediate Persist: Save partial progress
                db_manager.upsert_data(pd.DataFrame(new_records))

                # Reset batch
                batch_images = []
                batch_meta = []

    return pd.DataFrame(results)


def _run_clustering(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Standard Agglomerative Clustering per WSI.
    """
    final_dfs = []
    single_dfs = []

    if df.empty:
        return df

    grouped = df.groupby("wsi_id")
    for wsi_id, group in grouped:
        wsi_df = group.copy()

        if len(wsi_df) < 2:
            wsi_df["cluster_id"] = 0
            final_dfs.append(wsi_df)
            single_dfs.append(wsi_df)
            continue

        vectors = np.vstack(wsi_df["vector"].values)
        vectors_norm = normalize(vectors, axis=1, norm="l2")

        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="average"
        )
        wsi_df["cluster_id"] = model.fit_predict(vectors_norm)
        final_dfs.append(wsi_df)

    return pd.concat(final_dfs, ignore_index=True), (
        pd.concat(single_dfs, ignore_index=True) if len(single_dfs) > 0 else None
    )


# =============================================================================
# 4. Endpoints
# =============================================================================


@app.post("/v1/cluster/from_directory", response_class=JSONResponse)
async def cluster_from_directory(request: DirectoryRequest = Body(...)):
    """
    1. Scan Dir
    2. Smart Extract (Cache + Compute)
    3. Cluster
    4. Upsert Results
    """
    try:
        search_root = os.path.join(DATA_DIR, request.path)
        app_logger.info(f"Scanning {search_root}...")

        image_paths = []
        for ext in ["*.png", "*.jpg", "*.tif"]:
            image_paths.extend(glob(os.path.join(search_root, "**", ext), recursive=True))

        if not image_paths:
            raise HTTPException(404, "No images found.")

        # 1. Smart Extraction
        df_data = _smart_extract_features(app.state.feature_extractor, image_paths)

        # 2. Clustering
        threshold = request.threshold or 0.35
        clustered_df, _ = _run_clustering(df_data, threshold)

        # 3. Save to DB (Upsert)
        db_manager.upsert_data(clustered_df)

        # 4. Format Response
        results = []
        for wsi_id, w_group in clustered_df.groupby("wsi_id"):
            clusters = []
            for c_id, c_group in w_group.groupby("cluster_id"):
                clusters.append(
                    {"cluster_id": int(c_id), "count": len(c_group), "slides": c_group["slide_id"].tolist()}
                )
            results.append({"wsi_id": wsi_id, "clusters_count:": len(clusters), "clusters_details": clusters})

        output = {"id": str(uuid.uuid4()), "data": results}
        app_logger.info(f"Clustering done for {search_root}, result = {output}")

        return output

    except Exception as e:
        app_logger.error(f"Cluster Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/find_similar_nodes")
async def find_similar_nodes(request: SimilarityRequest = Body(...)):
    """
    New Feature: Find morphologically similar nodes across the database.
    """
    try:
        # 1. Get the source vector from DB
        source_df = db_manager.get_existing_vectors([request.slide_id])
        if source_df.empty:
            raise HTTPException(404, f"Slide ID {request.slide_id} not found in database. Run clustering first.")

        query_vector = source_df.iloc[0]["vector"]

        # 2. Global Search
        similar_df = db_manager.search_similarity(
            query_vector, limit=request.limit + 1, exclude_id=request.slide_id  # Fetch extra to account for self-match
        )

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

        return {"query_id": request.slide_id, "matches": matches}

    except Exception as e:
        app_logger.error(f"Search Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/tune_parameters")
async def tune_parameters(request: TuningRequest = Body(...)):
    """
    Uses Smart Extract to speed up tuning iterations.
    """
    try:
        # Load Ground Truth
        search_root = os.path.join(DATA_DIR, request.path)

        app_logger.info(f"Parameter Tuning: scanning {search_root}...")

        gt_path = os.path.join(search_root, request.ground_truth_csv)
        if not os.path.exists(gt_path):
            raise HTTPException(404, "Ground truth CSV not found.")

        gt_df = pd.read_csv(gt_path)
        gt_df.columns = [c.strip() for c in gt_df.columns]
        ground_truth_map = dict(zip(gt_df["image_name"], gt_df["lymph_node_total_num"]))

        # Scan & Extract
        image_paths = []
        for ext in ["*.png"]:
            image_paths.extend(glob(os.path.join(search_root, "**", ext), recursive=True))

        # Filter for only those in Ground Truth to save time
        relevant_paths = [p for p in image_paths if os.path.basename(os.path.dirname(p)) in ground_truth_map]

        # Retrieve/Compute Vectors
        data_df = _smart_extract_features(app.state.feature_extractor, relevant_paths)

        # Run Simulation (Logic borrowed from your original code)
        test_thresholds = np.arange(0.1, 1.05, 0.05).tolist()
        results = []

        for thresh in test_thresholds:
            temp_df, single_slide_df = _run_clustering(data_df, threshold=thresh)
            total_error = 0
            wsi_count = 0
            hit_count = 0

            for wsi_id, group in temp_df.groupby("wsi_id"):
                if wsi_id in ground_truth_map:
                    pred_k = group["cluster_id"].nunique()
                    actual_k = ground_truth_map[wsi_id]

                    # Data Error in Label, if there is just one slide, it must be one cluster
                    if wsi_id in single_slide_df["wsi_id"].tolist():
                        if pred_k == 1:
                            actual_k = 1

                    abs_error = abs(pred_k - actual_k)

                    if abs_error == 0:
                        hit_count += 1

                    total_error += abs_error
                    wsi_count += 1

            if wsi_count > 0:
                results.append(
                    {
                        "threshold": round(thresh, 2),
                        "accuracy": round(hit_count / wsi_count, 3),
                        "mae": round(total_error / wsi_count, 3),
                        "hit_count": hit_count,
                        "wsi_count": wsi_count,
                    }
                )

        results.sort(key=lambda x: x["mae"])
        output = {"best_parameter": results[0], "all_results": results}

        app_logger.info(f"Parameter Tuning done for {search_root}, best result = {output["best_parameter"]}")
        return output

    except Exception as e:
        app_logger.error(f"Tuning Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


def start_server():
    """Entry point for the 'histology-api' command."""
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=DEBUG_MODE,  # Reload must be False for production scripts
    )


if __name__ == "__main__":
    start_server()
