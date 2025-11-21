import numpy as np
import os
import json
import time
import uuid
import logging
from glob import glob
from typing import List, Dict, Optional, Any

# --- API & Server ---
import uvicorn
from fastapi import FastAPI, Body, HTTPException
from starlette.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- ML & Data ---
import lancedb
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# --- Imports for Hybrid Support ---
# We import both, but we will only load the heavy weights for the selected one.
import torch
import timm
from torchvision import transforms as T
from PIL import Image

# TensorFlow (Suppress warnings if just using PyTorch, but need import for ResNet)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tensorflow
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as tf_image

load_dotenv()

# =============================================================================
# Configuration & Environment
# =============================================================================

# --- SWITCH ---
# Options: "resnet50" or "uni2-h"
SELECTED_MODEL = os.getenv("SELECTED_MODEL", "uni2-h").lower()

# --- Paths ---
DATA_DIR = os.getenv("DATA_DIR", r"D:/data/ruijin/Data/crops")
UNI_LOCAL_PATH = os.getenv("UNI_LOCAL_PATH", r"D:/models/huggingface/hub/uni2-h/pytorch_model.bin")
RESNET50_LOCAL_PATH = os.getenv(
    "RESNET50_LOCAL_PATH", r"D:/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Server Config ---
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# --- Clustering Config ---
DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.35"))
# Note: UNI uses 224x224. ResNet often uses 224x224 as well.
IMAGE_TARGET_SIZE = tuple(map(int, os.getenv("IMAGE_TARGET_SIZE", "224,224").split(",")))
SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", "*.png,*.jpg,*.tif").split(",")
DEFAULT_DB_PATH = os.getenv("DEFAULT_DB_PATH", "lancedb_histology")
DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "histology_specimens")

# =============================================================================
# Logging Setup
# =============================================================================
logging.getLogger("uvicorn.access").disabled = True
app_logger = logging.getLogger("api.application")
app_logger.setLevel(logging.INFO)
app_logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8")
file_handler.setFormatter(formatter)

app_logger.addHandler(console_handler)
app_logger.addHandler(file_handler)
app_logger.propagate = False

# =============================================================================
# 1. Model Abstraction (Strategy Pattern)
# =============================================================================


class BaseExtractor:
    """Interface for feature extraction."""

    def load(self):
        raise NotImplementedError

    def extract(self, img_path: str) -> List[float]:
        raise NotImplementedError


class ResNet50Extractor(BaseExtractor):
    """TensorFlow / Keras Implementation"""

    def __init__(self, weights_path):
        self.model = None
        self.weights_path = weights_path

    def load(self):
        app_logger.info("Loading TensorFlow ResNet50 (ImageNet)...")
        # Using standard Keras ResNet50
        # Check if local weights file exists
        if os.path.exists(self.weights_path):
            app_logger.info(f"Loading ResNet50 weights from local file: {self.weights_path}")
            self.model = ResNet50(
                weights=self.weights_path, include_top=False, input_shape=(*IMAGE_TARGET_SIZE, 3), pooling="avg"
            )
        else:
            app_logger.warning(f"Local ResNet50 weights not found at {self.weights_path}, using 'imagenet'")
            self.model = ResNet50(
                weights="imagenet", include_top=False, input_shape=(*IMAGE_TARGET_SIZE, 3), pooling="avg"
            )

        app_logger.info("ResNet50 loaded.")

    def extract(self, img_path: str) -> List[float]:
        try:
            img = tf_image.load_img(img_path, target_size=IMAGE_TARGET_SIZE)
            x = tf_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            return self.model.predict(x, verbose=0).flatten().tolist()
        except Exception as e:
            app_logger.warning(f"[ResNet] Error processing {img_path}: {e}")
            return []


class Uni2hExtractor(BaseExtractor):
    """PyTorch / ViT-H Implementation"""

    def __init__(self, weights_path):
        self.model = None
        self.transform = None
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.weights_path = weights_path

    def load(self):
        app_logger.info(f"Loading UNI2-h (ViT-Huge) on {self.device}...")

        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"UNI2-h weights not found at {self.weights_path}")

        # # Create Architecture
        # self.model = timm.create_model(
        #     "vit_giant_patch14_dinov2", pretrained=False, init_values=1e-5, num_classes=0, dynamic_img_size=True
        # )
        # # Load State Dict
        # state_dict = torch.load(self.weights_path, map_location="cpu")
        # Fix dictionary keys if they contain 'module.' prefix
        # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # self.model.load_state_dict(new_state_dict, strict=False)
        timm_kwargs = {
            "model_name": "vit_giant_patch14_224",
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = timm.create_model(pretrained=False, **timm_kwargs)
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu"), strict=True)

        self.model.to(self.device)
        self.model.eval()

        # Define Transform (ImageNet Normalization)
        self.transform = T.Compose(
            [
                T.Resize(IMAGE_TARGET_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        app_logger.info("UNI2-h loaded.")

    def extract(self, img_path: str) -> List[float]:
        try:
            img = Image.open(img_path).convert("RGB")
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self.model(input_tensor)
                # ViT-H usually returns 1024-dim (or 1280 depending on head)
                return features.cpu().numpy().flatten().tolist()
        except Exception as e:
            app_logger.warning(f"[UNI] Error processing {img_path}: {e}")
            return []


# =============================================================================
# 2. API & Logic
# =============================================================================


class TuningRequest(BaseModel):
    path: str = ""
    ground_truth_csv: str


class DirectoryRequest(BaseModel):
    path: str = ""
    db_path: str = DEFAULT_DB_PATH
    table_name: str = DEFAULT_TABLE_NAME
    threshold: Optional[float] = DEFAULT_DISTANCE_THRESHOLD


app = FastAPI(title="Histology Clustering API (Hybrid)", version="2.1.0")

# Global Store for the active model
extractor: Optional[BaseExtractor] = None


@app.on_event("startup")
def startup_event():
    global extractor
    app_logger.info(f"Starting Server. Selected Model: {SELECTED_MODEL.upper()}")

    if SELECTED_MODEL == "resnet50":
        extractor = ResNet50Extractor(weights_path=RESNET50_LOCAL_PATH)
    elif SELECTED_MODEL == "uni2-h":
        extractor = Uni2hExtractor(weights_path=UNI_LOCAL_PATH)
    else:
        raise ValueError(f"Unknown model selected: {SELECTED_MODEL}. Use 'resnet50' or 'uni2-h'")

    extractor.load()


# =============================================================================
# 3. Shared Helper Functions
# =============================================================================


def _run_clustering(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    final_dfs = []
    single_vector = []
    grouped = df.groupby("wsi_id")

    for wsi_id, group in grouped:
        wsi_df = group.copy()
        vectors = np.array(wsi_df["vector"].tolist())

        # L2 Normalization is crucial for both models to use Euclidean distance like Cosine
        vectors_norm = normalize(vectors, axis=1, norm="l2")

        if len(vectors) < 2:
            wsi_df["cluster_id"] = 0
            single_vector.append(wsi_id)
        else:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=threshold, metric="euclidean", linkage="average"
            )
            wsi_df["cluster_id"] = model.fit_predict(vectors_norm)
        final_dfs.append(wsi_df)

    if not final_dfs:
        return df, single_vector
    return pd.concat(final_dfs, ignore_index=True), single_vector


def _run_tuning_simulation(data_df: pd.DataFrame, ground_truth: Dict[str, int]):
    test_thresholds = np.arange(0.1, 1.05, 0.05).tolist()
    results = []
    errors = []

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

            if wsi_id in single_slides:
                single_crop_count += 1
                if abs_error > 0:
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
                "match_count": match_count,
                "wsi_count": wsi_count,
                "single_crop_count": single_crop_count,
                "error_count": error_count,
            }
        )

    results.sort(key=lambda x: x["accuracy"], reverse=True)
    return results[0], results, list(set(errors))


# =============================================================================
# 4. Endpoints
# =============================================================================


@app.post("/v1/tune_parameters", response_class=JSONResponse)
async def tune_parameters(request: TuningRequest = Body(...)):
    global extractor
    try:
        search_root = os.path.join(DATA_DIR, request.path)
        label_file = os.path.join(search_root, request.ground_truth_csv)
        gt_df = pd.read_csv(label_file)
        gt_df.columns = [c.strip() for c in gt_df.columns]
        ground_truth_map = dict(zip(gt_df.iloc[:, 0], gt_df.iloc[:, 1]))

        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(search_root, "**", ext.strip())
            image_paths.extend(glob(pattern, recursive=True))

        if not image_paths:
            raise HTTPException(404, "No images found.")

        all_data = []
        app_logger.info(f"Processing {len(image_paths)} images with {SELECTED_MODEL}...")

        for i, p in enumerate(image_paths):
            if i % 10 == 0:
                app_logger.info(f"Extracted {i}/{len(image_paths)}")

            fname = os.path.basename(p)
            wsi_id = os.path.basename(os.path.dirname(p))

            if wsi_id in ground_truth_map:
                # POLYMORPHIC CALL - Doesn't matter which model is loaded
                vec = extractor.extract(p)
                if vec:
                    all_data.append({"file_name": fname, "wsi_id": wsi_id, "slide_id": fname, "vector": vec})

        if not all_data:
            raise HTTPException(404, "No vectors extracted.")

        app_logger.info(f"Tuning parameters for {len(image_paths)} images with {SELECTED_MODEL}...")
        best_param, all_results, errors = _run_tuning_simulation(pd.DataFrame(all_data), ground_truth_map)

        results = {
            "model_used": SELECTED_MODEL,
            "recommendation": best_param,
            "details": all_results,
            "error_list": errors,
        }

        app_logger.info(f"=" * 64)
        app_logger.info(f"Done on tune_parameters, with results: {results}")
        app_logger.info(f"=" * 64)

        return results

    except Exception as e:
        app_logger.error(f"Tune Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/v1/cluster/from_directory", response_class=JSONResponse)
async def cluster_from_directory(request: DirectoryRequest = Body(...)):
    global extractor
    try:
        search_root = os.path.join(DATA_DIR, request.path)
        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(search_root, "**", ext.strip())
            image_paths.extend(glob(pattern, recursive=True))

        if not image_paths:
            raise HTTPException(404, "No images found.")

        all_data = []
        app_logger.info(f"Processing {len(image_paths)} images with {SELECTED_MODEL}...")

        for i, p in enumerate(image_paths):
            if i % 20 == 0:
                app_logger.info(f"Extracted {i}/{len(image_paths)}")

            fname = os.path.basename(p)
            wsi_id = os.path.basename(os.path.dirname(p))

            # POLYMORPHIC CALL
            vec = extractor.extract(p)
            if vec:
                all_data.append({"file_name": fname, "wsi_id": wsi_id, "slide_id": fname, "vector": vec})

        if not all_data:
            raise HTTPException(400, "No valid images processed.")

        active_thresh = request.threshold if request.threshold is not None else DEFAULT_DISTANCE_THRESHOLD
        clustered_df, _ = _run_clustering(pd.DataFrame(all_data), threshold=active_thresh)

        # Save DB
        db = lancedb.connect(request.db_path)
        try:
            db.drop_table(request.table_name)
        except:
            pass
        db.create_table(request.table_name, data=clustered_df)

        # Format Response
        results = []
        for wsi_id, w_group in clustered_df.groupby("wsi_id"):
            clusters = []
            for c_id, c_group in w_group.groupby("cluster_id"):
                clusters.append({"cluster_id": int(c_id), "slides": c_group["file_name"].tolist()})
            results.append({"wsi_id": wsi_id, "cluster_count": len(clusters), "clusters": clusters})

        results = JSONResponse(
            content={
                "id": str(uuid.uuid4()),
                "model": SELECTED_MODEL,
                "choices": [{"message": {"content": json.dumps(results)}}],
            }
        )

        app_logger.info(f"=" * 64)
        app_logger.info(f"Done on cluster_from_directory, with results: {results}")
        app_logger.info(f"=" * 64)

        return results

    except Exception as e:
        app_logger.error(f"Cluster Error: {e}", exc_info=True)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=DEBUG)
