#!/bin/bash
/opt/conda/envs/app_env/bin/python -m uvicorn lymphaclustering.app:app --host 0.0.0.0 --port 8000