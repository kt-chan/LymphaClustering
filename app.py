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
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- ML & Data ---
import lancedb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = os.getcwd()

# =============================================================================
# Logging Configuration
# =============================================================================

# --- 1. Application Logger (Replaces print statements) ---
# This logger is for application-specific events (e.g., "Model loaded")
app_logger = logging.getLogger("api.application")
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
app_logger.addHandler(app_handler)
app_logger.propagate = False

# --- 2. Access Logger (Apache Combined Log Format) ---
# This logger is specifically for web requests, formatted like Apache's logs
access_logger = logging.getLogger("api.access")
access_logger.setLevel(logging.INFO)
access_handler = logging.StreamHandler()
access_handler.setFormatter(logging.Formatter("%(message)s")) # We format the message ourselves
access_logger.addHandler(access_handler)
access_logger.propagate = False

class ApacheLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log requests in Apache Combined Log Format.
    Format: host ident authuser [timestamp] "request" status bytes "referer" "user-agent"
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Log generation
        process_time = (time.time() - start_time) * 1000 # in milliseconds
        
        # 1. Host (%h)
        host = request.client.host if request.client else "-"
        
        # 2. Ident (%l) - Not available
        ident = "-"
        
        # 3. Authuser (%u) - Not available
        authuser = "-"
        
        # 4. Timestamp (%t)
        now = datetime.now()
        timestamp = now.strftime('[%d/%b/%Y:%H:%M:%S %z]')
        
        # 5. Request line ("%r")
        request_line = f'"{request.method} {request.url.path}'
        if request.url.query:
            request_line += f'?{request.url.query}'
        request_line += f' {request.scope["http_version"]}"'
        
        # 6. Status code (%>s)
        status_code = response.status_code
        
        # 7. Response size (%b)
        response_size = response.headers.get("content-length", "-")

        # 8. Referer
        referer = request.headers.get("referer", "-")
        
        # 9. User-Agent
        user_agent = request.headers.get("user-agent", "-")

        # Combine them
        log_message = f'{host} {ident} {authuser} {timestamp} {request_line} {status_code} {response_size} "{referer}" "{user_agent}"'
        
        access_logger.info(log_message)
        
        return response

# =============================================================================
# Environment Variables Configuration
# =============================================================================

# Server configuration
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
if DEBUG:
    app_logger.setLevel(logging.DEBUG)

# Database configuration
DEFAULT_DB_PATH = os.getenv("DEFAULT_DB_PATH", "lancedb_histology")
DEFAULT_TABLE_NAME = os.getenv("DEFAULT_TABLE_NAME", "histology_specimens")

# Model configuration
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "imagenet")
MODEL_INCLUDE_TOP = os.getenv("MODEL_INCLUDE_TOP", "False").lower() == "true"
MODEL_POOLING = os.getenv("MODEL_POOLING", "avg")

# Clustering configuration
MAX_CLUSTERS = int(os.getenv("MAX_CLUSTERS", "10"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
N_INIT = int(os.getenv("N_INIT", "10"))

# Image processing configuration
IMAGE_TARGET_SIZE = tuple(map(int, os.getenv("IMAGE_TARGET_SIZE", "224,224").split(',')))
SUPPORTED_EXTENSIONS = os.getenv("SUPPORTED_EXTENSIONS", "*.png").split(',')

# =============================================================================
# 1. API Models (Request & Response)
# =============================================================================

class DirectoryRequest(BaseModel):
    """Pydantic model for the directory processing request."""
    path: str
    db_path: str = DEFAULT_DB_PATH
    table_name: str = DEFAULT_TABLE_NAME

class OpenAIChoice(BaseModel):
    """OpenAI-style choice object."""
    index: int
    message: Dict[str, Any]
    finish_reason: str = "completed"

class OpenAIUsage(BaseModel):
    """OpenAI-style usage object."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class OpenAIResponse(BaseModel):
    """OpenAI-compatible response model for clustering."""
    id: str
    object: str = "clustering.result"
    created: int
    model: str = "resnet50-kmeans-lancedb"
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

# =============================================================================
# 2. FastAPI App & Model-Loading
# =============================================================================

app = FastAPI(
    title="Histology Clustering API",
    description="API for clustering WSI slides using ResNet50 and LanceDB",
    version="1.0.0"
)

# Add the custom Apache-style logging middleware
app.add_middleware(ApacheLoggingMiddleware)

# Use app.state to store the model, so it's loaded only once.
app.state.feature_extractor = None

@app.on_event("startup")
def load_model():
    """Load the ML model into memory at application startup."""
    # --- ADD THESE LINES ---
    app_logger.info(f"Starting API server at http://{HOST}:{PORT}")
    app_logger.info(f"View API docs at http://{HOST}:{PORT}/docs")
    app_logger.info(f"Debug mode: {DEBUG}")
    app_logger.info("Loading ResNet50 model...")
    app_logger.info(f"Model configuration: weights={MODEL_WEIGHTS}, include_top={MODEL_INCLUDE_TOP}, pooling={MODEL_POOLING}")
    
    base_model = ResNet50(
        weights=MODEL_WEIGHTS, 
        include_top=MODEL_INCLUDE_TOP, 
        input_shape=(*IMAGE_TARGET_SIZE, 3), 
        pooling=MODEL_POOLING
    )
    app.state.feature_extractor = base_model
    app_logger.info("Model loaded successfully.")

# =============================================================================
# 3. Core Service Logic (Refactored from your script)
# =============================================================================

# Define the regex pattern for parsing filenames
FILENAME_PATTERN = re.compile(r"(\S+)#(\S+)\.png$") 

def _parse_filename(filename: str) -> Optional[tuple]:
    """Parses WSI_ID and Slide_ID from a filename."""
    match = FILENAME_PATTERN.search(filename)
    if not match:
        app_logger.warning(f"Skipping {filename}: Does not match WSI_ID#Slide_ID.png pattern.")
        return None
    return match.group(1), match.group(2) # (wsi_id, slide_id)

def _preprocess_image_from_bytes(img_bytes: bytes) -> np.ndarray:
    """Preprocesses image data from in-memory bytes."""
    img = image.load_img(io.BytesIO(img_bytes), target_size=IMAGE_TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def _preprocess_image_from_path(img_path: str) -> Optional[np.ndarray]:
    """Preprocesses image data from a file path."""
    try:
        img = image.load_img(img_path, target_size=IMAGE_TARGET_SIZE)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array_expanded)
    except FileNotFoundError:
        app_logger.error(f"Error: Image file not found at {img_path}")
        return None
    except Exception as e:
        app_logger.error(f"Error loading image {img_path}: {e}")
        return None

def _extract_vector(model: Model, preprocessed_img: np.ndarray) -> List[float]:
    """Extracts a feature vector using the loaded model."""
    features = model.predict(preprocessed_img, verbose=0).flatten()
    return features.tolist()

def _run_clustering(data_df: pd.DataFrame) -> pd.DataFrame:
    """Performs K-Means clustering and returns the DataFrame with cluster_id."""
    features_array = np.array(data_df['vector'].tolist())
    
    app_logger.info("--- Clustering: Finding Optimal k ---")
    max_k = min(len(features_array) - 1, MAX_CLUSTERS) # Limit k for robustness
    
    if max_k < 2:
        app_logger.warning("Not enough samples for clustering. Setting k=1.")
        final_labels = np.zeros(len(features_array), dtype=int)
    else:
        silhouette_scores = {}
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
            cluster_labels = kmeans.fit_predict(features_array)
            score = silhouette_score(features_array, cluster_labels)
            silhouette_scores[k] = score
            app_logger.debug(f"Silhouette score for k={k}: {score}")
            
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        app_logger.info(f"Best k (highest silhouette score) is: {best_k}")
        
        final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=N_INIT)
        final_labels = final_kmeans.fit_predict(features_array)

    data_df['cluster_id'] = final_labels
    return data_df

def _save_to_lancedb(data_df: pd.DataFrame, db_path: str, table_name: str) -> int:
    """Saves the DataFrame to a LanceDB table, dropping if it exists."""
    app_logger.info(f"--- Writing Vectors and Metadata to LanceDB '{table_name}' ---")
    db = lancedb.connect(db_path)
    
    try:
        db.drop_table(table_name)
        app_logger.info(f"Dropped existing table '{table_name}'.")
    except Exception:
        app_logger.info(f"No existing table '{table_name}' to drop.")
        
    tbl = db.create_table(table_name, data=data_df)
    count = tbl.count_rows()
    app_logger.info(f"Successfully created table with {count} records.")
    return count

def _format_json_output(data_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generates the final JSON output structure."""
    grouped = data_df.groupby(['wsi_id', 'cluster_id']).agg(
        file_names=('file_name', list),
        slide_ids=('slide_id', list)
    ).reset_index()
    
    results = []
    for _, row in grouped.iterrows():
        results.append({
            "WSI_ID": row['wsi_id'],
            "cluster_id": int(row['cluster_id']),
            "count_of_slide_ids": len(row['slide_ids']),
            "list_of_Slide_IDs": row['slide_ids'],
            "list_of_file_names": row['file_names']
        })
    
    # Calculate the total number of unique clusters found
    cluster_count = data_df['cluster_id'].nunique()
    
    # Return a dictionary containing both the count and the results list
    return {
        "cluster_count": cluster_count,
        "cluster_output": results
    }

def _create_openai_response(data: List[Dict[str, Any]]) -> JSONResponse:
    """Wraps the final data in an OpenAI-compatible response format."""
    response_id = f"cluster-{uuid.uuid4()}"
    created_time = int(time.time())
    
    # Wrap the JSON data inside the 'content' field
    content_data = {"clustering_results": data}
    
    response_model = OpenAIResponse(
        id=response_id,
        created=created_time,
        choices=[
            OpenAIChoice(
                index=0,
                message={"role": "assistant", "content": json.dumps(content_data)},
                finish_reason = "success"
            )
        ],
        usage=OpenAIUsage() # Dummy usage
    )
    return JSONResponse(content=response_model.dict())

# =============================================================================
# 4. API Endpoints
# =============================================================================

@app.post("/v1/cluster/from_directory", response_model=OpenAIResponse)
async def cluster_from_directory(request: DirectoryRequest = Body(...)):
    """
    Processes images from a server-side directory, clusters them,
    and saves them to LanceDB.
    """
    try:
        start_time = time.time()
        app_logger.info(f"Starting /v1/cluster/from_directory for path: {request.path}")
        full_data_path = os.path.join(PROJECT_ROOT, request.path)
        
        # Use environment variable for file extensions
        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            search_pattern = os.path.join(full_data_path, ext.strip())
            image_paths.extend(glob(search_pattern, recursive=True))
            
        if not image_paths:
            app_logger.error(f"No images found in directory: {request.path}")
            raise HTTPException(
                status_code=404, 
                detail=f"No images matching patterns {SUPPORTED_EXTENSIONS} found in directory: {request.path}"
            )
        
        app_logger.info(f"Found {len(image_paths)} images to process.")
        
        all_data = []
        model = app.state.feature_extractor
        
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            parsed_ids = _parse_filename(filename)
            if not parsed_ids:
                continue

            preprocessed_img = _preprocess_image_from_path(img_path)
            if preprocessed_img is None:
                continue
                
            vector = _extract_vector(model, preprocessed_img)
            all_data.append({
                "file_name": filename,
                "wsi_id": parsed_ids[0],
                "slide_id": parsed_ids[1],
                "vector": vector
            })
            
        if not all_data:
            app_logger.error("No valid images could be processed from the directory.")
            raise HTTPException(status_code=400, detail="No valid images could be processed.")

        app_logger.info(f"Successfully processed {len(all_data)} valid images.")
        
        data_df = pd.DataFrame(all_data)
        clustered_df = _run_clustering(data_df)
        _save_to_lancedb(clustered_df, request.db_path, request.table_name)
        json_output = _format_json_output(clustered_df)
        
        end_time = time.time()
        app_logger.info(f"Finished /v1/cluster/from_directory in {end_time - start_time:.2f} seconds.")
        
        return _create_openai_response(json_output)
        
    except HTTPException as he:
        # Re-raise HTTPException so FastAPI handles it
        raise he
    except Exception as e:
        app_logger.error(f"An unexpected error occurred in /from_directory: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/v1/cluster/from_upload", response_model=OpenAIResponse)
async def cluster_from_upload(
    files: List[UploadFile] = File(..., description="List of PNG images (WSI_ID#Slide_ID.png)"),
    db_path: str = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME
):
    """
    Processes images from a binary upload, clusters them,
    and saves them to LanceDB.
    """
    try:
        start_time = time.time()
        app_logger.info(f"Starting /v1/cluster/from_upload with {len(files)} files.")
        
        if not files:
            app_logger.warning("No files were uploaded to /from_upload endpoint.")
            raise HTTPException(status_code=400, detail="No files were uploaded.")
        
        all_data = []
        model = app.state.feature_extractor
        
        for file in files:
            filename = file.filename
            parsed_ids = _parse_filename(filename)
            if not parsed_ids:
                continue
                
            file_bytes = await file.read()
            preprocessed_img = _preprocess_image_from_bytes(file_bytes)
            vector = _extract_vector(model, preprocessed_img)
            
            all_data.append({
                "file_name": filename,
                "wsi_id": parsed_ids[0],
                "slide_id": parsed_ids[1],
                "vector": vector
            })

        if not all_data:
            app_logger.error("No valid images could be processed from the upload.")
            raise HTTPException(status_code=400, detail="No valid images could be processed from the upload.")

        app_logger.info(f"Successfully processed {len(all_data)} valid images from upload.")

        data_df = pd.DataFrame(all_data)
        clustered_df = _run_clustering(data_df)
        _save_to_lancedb(clustered_df, db_path, table_name)
        json_output = _format_json_output(clustered_df)
        
        end_time = time.time()
        app_logger.info(f"Finished /v1/cluster/from_upload in {end_time - start_time:.2f} seconds.")
        
        return _create_openai_response(json_output)
        
    except HTTPException as he:
        # Re-raise HTTPException so FastAPI handles it
        raise he
    except Exception as e:
        app_logger.error(f"An unexpected error occurred in /from_upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# =============================================================================
# 5. Run the Server
# =============================================================================

if __name__ == "__main__":
    """Run the API server with Uvicorn."""
    app_logger.info(f"Starting API server at http://{HOST}:{PORT}")
    app_logger.info(f"View API docs at http://{HOST}:{PORT}/docs")
    app_logger.info(f"Debug mode: {DEBUG}")
    
    uvicorn.run(
        "__main__:app", # Use "__main__:app" when running script directly
        host=HOST, 
        port=PORT, 
        reload=DEBUG,
        log_config=None # Disable uvicorn's default access logging
    )

