#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate ./venv
python -m uvicorn app:main