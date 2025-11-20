import numpy as np
import os
import json
import re
import io
import time
import uuid
import logging
from datetime import datetime
from glob import glob
from typing import List, Dict, Any, Optional

# --- API & Server ---
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Form
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- ML & Data ---
import lancedb
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from dotenv import load_dotenv

load_dotenv()

# UPDATED: New Data Directory
DATA_DIR = os.getenv("DATA_DIR", r"D:\data\ruijin\Data\crops")

# =============================================================================
# Logging Configuration
# =============================================================================
# Disable uvicorn default logging to avoid duplicates
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
LOG_FILENAME = f"app.log"
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILENAME)

# Configure the application logger
app_logger = logging.getLogger("api.application")
app_logger.setLevel(logging.INFO)

# Clear any existing handlers to avoid duplicates
app_logger.handlers.clear()

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console Handler (keeps your existing console output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
app_logger.addHandler(console_handler)

# File Handler (new - logs to file)
file_handler = logging.FileHandler(LOG_FILEPATH, encoding="utf-8")
file_handler.setFormatter(formatter)
app_logger.addHandler(file_handler)

# Prevent propagation to root logger to avoid double logging
app_logger.propagate = False

app_logger.info(f"Logging initialized. Console and file logging enabled.")
app_logger.info(f"Log file: {LOG_FILEPATH}")

# =============================================================================
# Environment Variables
# =============================================================================
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.35"))

IMAGE_TARGET_SIZE = tuple(map(int, os.getenv("IMAGE_TARGET_SIZE", "224,224").split(",")))
SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", "*.png").split(",")
DEFAULT_DB_PATH = os.getenv("DEFAULT_DB_PATH", "lancedb_histology")
DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "histology_specimens")

MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "imagenet")
MODEL_INCLUDE_TOP = os.getenv("MODEL_INCLUDE_TOP", "False").lower() == "true"
MODEL_POOLING = os.getenv("MODEL_POOLING", "avg")


# =============================================================================
# 1. Models
# =============================================================================


class TuningRequest(BaseModel):
    path: str = ""  # Default to empty string to scan the whole DATA_DIR
    ground_truth_csv: str


class DirectoryRequest(BaseModel):
    path: str = ""  # Default to empty string
    db_path: str = DEFAULT_DB_PATH
    table_name: str = DEFAULT_TABLE_NAME
    threshold: Optional[float] = DEFAULT_DISTANCE_THRESHOLD


# =============================================================================
# 2. App & Model Loading
# =============================================================================

app = FastAPI(title="Histology Clustering API", version="1.3.0")
app.state.feature_extractor = None


@app.on_event("startup")
def load_model():
    app_logger.info(f"Starting API server. Data Root: {DATA_DIR}")
    app_logger.info("Loading ResNet50 model...")
    base_model = ResNet50(
        weights=MODEL_WEIGHTS, include_top=MODEL_INCLUDE_TOP, input_shape=(*IMAGE_TARGET_SIZE, 3), pooling=MODEL_POOLING
    )
    app.state.feature_extractor = base_model
    app_logger.info("Model loaded successfully.")


# =============================================================================
# 3. Core Logic (Updated Parsing)
# =============================================================================


def _extract_vector(model: Model, img_path: str) -> List[float]:
    try:
        img = image.load_img(img_path, target_size=IMAGE_TARGET_SIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x, verbose=0).flatten().tolist()
    except Exception as e:
        app_logger.warning(f"Failed to load {img_path}: {e}")
        return []


def _run_clustering(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    final_dfs = []
    single_vector = []
    # Group by WSI ID so we cluster only within the same patient/case
    grouped = df.groupby("wsi_id")

    for wsi_id, group in grouped:
        wsi_df = group.copy()
        vectors = np.array(wsi_df["vector"].tolist())

        # Normalize for Cosine-like Euclidean distance
        vectors_norm = normalize(vectors, axis=1, norm="l2")

        if len(vectors) < 2:
            wsi_df["cluster_id"] = 0
            single_vector.append(wsi_id)
        else:
            # Agglomerative Clustering with Dynamic Threshold
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="average"
            )
            wsi_df["cluster_id"] = model.fit_predict(vectors_norm)

        final_dfs.append(wsi_df)

    if not final_dfs:
        return df, single_vector
    return pd.concat(final_dfs, ignore_index=True), single_vector


def _run_tuning_simulation(data_df: pd.DataFrame, ground_truth: Dict[str, int]):
    # Test range: 0.2 (Strict) -> 1.0 (Loose)
    test_thresholds = np.arange(0.1, 1.05, 0.05).tolist()
    results = []
    errors = []

    app_logger.info(f"--- Starting Tuning on {len(ground_truth)} labeled WSIs ---")

    for thresh in test_thresholds:
        total_absolute_error = 0
        match_count = 0
        wsi_count = 0
        single_crop_count = 0
        error_count = 0

        clustered_df, single_slides = _run_clustering(data_df, threshold=thresh)

        for wsi_id, predicted_group in clustered_df.groupby("wsi_id"):
            if wsi_id not in ground_truth:
                continue

            predicted_k = predicted_group["cluster_id"].nunique()
            actual_k = ground_truth[wsi_id]
            abs_error = abs(predicted_k - actual_k)

            #  Check for single slide of a wsi
            if wsi_id in single_slides:
                single_crop_count += 1
                if abs_error > 0:
                    # app_logger.error(f"Critical Failure for Single Slide on WSI: {wsi_id}")
                    error_count += 1
                    errors.append(wsi_id)
                    continue

            total_absolute_error += abs_error

            if abs_error == 0:
                match_count += 1

            wsi_count += 1

        if wsi_count == 0:
            continue

        mae = total_absolute_error / wsi_count
        results.append(
            {
                "threshold": f"{thresh:.2f}",
                "accuracy": match_count / wsi_count,
                "mae": round(mae, 3),
                "details": f"Avg error: {mae:.2f}",
                "match_count": match_count,
                "wsi_count": wsi_count,
                "single_crop_count": single_crop_count,
                "error_count": error_count,
            }
        )

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[0], results, list(set(errors))


# =============================================================================
# 4. API Endpoints
# =============================================================================


@app.post("/v1/tune_parameters", response_class=JSONResponse)
async def tune_parameters(request: TuningRequest = Body(...)):
    """
    Scans nested folders, matches against CSV Ground Truth, finds best Threshold.
    """
    try:
        start_time = time.time()
        app_logger.info("=" * 64)
        app_logger.info(f"Requesting tune_parameters on {start_time} ...")
        app_logger.info("=" * 64)
        # 1. Load Ground Truth
        search_root = os.path.join(DATA_DIR, request.path)
        label_file = os.path.join(search_root, request.ground_truth_csv)
        gt_df = pd.read_csv(label_file)
        gt_df.columns = [c.strip() for c in gt_df.columns]
        # Map WSI_ID -> Count
        ground_truth_map = dict(zip(gt_df.iloc[:, 0], gt_df.iloc[:, 1]))

        # 2. Find Images (Recursive Scan)
        # Structure: DATA_DIR / request.path / ** / *.png

        app_logger.info(f"Scanning for images in: {search_root}")

        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            # Recursive glob pattern for nested folders
            pattern = os.path.join(search_root, "**", ext.strip())
            found = glob(pattern, recursive=True)
            image_paths.extend(found)

        app_logger.info(f"Found {len(image_paths)} images total.")

        # 3. Extract Features
        all_data = []
        model = app.state.feature_extractor

        for i, p in enumerate(image_paths):
            if i % 10 == 0:
                app_logger.info(f"vectorizing images : {i}/{len(image_paths)}.")

            fname = os.path.basename(p)
            wsi_id = os.path.basename(os.path.dirname(p))
            slide_id = fname

            # Optimization: Only process if in Ground Truth
            if wsi_id in ground_truth_map:
                vec = _extract_vector(model, p)
                if vec:
                    all_data.append({"file_name": fname, "wsi_id": wsi_id, "slide_id": slide_id, "vector": vec})

        if not all_data:
            raise HTTPException(404, "No images matched the WSI IDs in your CSV.")

        data_df = pd.DataFrame(all_data)

        # 4. Run Simulation
        best_param, all_results, errors = _run_tuning_simulation(data_df, ground_truth_map)

        result = {
            "status": "success",
            "recommendation": {
                "best_threshold": best_param["threshold"],
                "accuracy": best_param["accuracy"],
                "mae": best_param["mae"],
                "match_count": best_param["match_count"],
                "wsi_count": best_param["wsi_count"],
                "single_crop_count": best_param["single_crop_count"],
                "corping_error_count": best_param["error_count"],
            },
            "details": all_results,
            "error_list": errors,
        }

        app_logger.info("=" * 64)
        app_logger.info(f"final results: {result}.")
        app_logger.info("=" * 64)

        return result

    except Exception as e:
        app_logger.error(f"Tuning Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/cluster/from_directory", response_class=JSONResponse)
async def cluster_from_directory(request: DirectoryRequest = Body(...)):
    """
    Production Endpoint. Recursive scan + Clustering.
    """
    try:
        # 1. Recursive Scan
        search_root = os.path.join(DATA_DIR, request.path)
        app_logger.info(f"Clustering images in: {search_root}")

        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(search_root, "**", ext.strip())
            image_paths.extend(glob(pattern, recursive=True))

        if not image_paths:
            raise HTTPException(404, "No images found.")

        # 2. Feature Extraction
        all_data = []
        model = app.state.feature_extractor

        for p in image_paths:
            fname = os.path.basename(p)
            wsi_id = os.path.basename(os.path.dirname(p))
            slide_id = fname
            vec = _extract_vector(model, p)
            if vec:
                all_data.append({"file_name": fname, "wsi_id": wsi_id, "slide_id": slide_id, "vector": vec})

        if not all_data:
            raise HTTPException(400, "No valid images processed.")

        # 3. Cluster
        active_thresh = request.threshold if request.threshold is not None else DEFAULT_DISTANCE_THRESHOLD
        clustered_df, single_slide = _run_clustering(pd.DataFrame(all_data), threshold=active_thresh)

        # 4. Save & Response
        db = lancedb.connect(request.db_path)
        try:
            db.drop_table(request.table_name)
        except:
            pass
        db.create_table(request.table_name, data=clustered_df)

        results = []
        for wsi_id, w_group in clustered_df.groupby("wsi_id"):
            clusters = []
            for c_id, c_group in w_group.groupby("cluster_id"):
                clusters.append({"cluster_id": int(c_id), "slides": c_group["file_name"].tolist()})
            results.append({"wsi_id": wsi_id, "cluster_count": len(clusters), "clusters": clusters})

        app_logger.info(f"cluster results: {results}.")

        return JSONResponse(
            content={"id": str(uuid.uuid4()), "choices": [{"message": {"content": json.dumps(results)}}]}
        )

    except Exception as e:
        app_logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=DEBUG)
