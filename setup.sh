#!/bin/bash

# --- Configuration ---
VENV_PATH="./venv"
PYTHON_VERSION="3.12"
# Set a Linux-appropriate cache path (e.g., in /tmp or user home)
UV_CACHE_DIR="$HOME/.cache/uv"

# Function to print error messages and exit
handle_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- 1. Check Dependencies ---

# Check if Conda is available (using 'command -v' which is robust)
if ! command -v conda &> /dev/null; then
    handle_error "Conda is not installed or not available in PATH."
fi

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    handle_error "requirements.txt not found in current directory."
fi

# --- 2. Install or Upgrade UV ---

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    # Ensure pip is run against the environment where you want uv installed (usually base or user)
    pip install uv || handle_error "Failed to install uv. Check if 'pip' is configured correctly."
else
    echo "Updating uv..."
    pip install --upgrade uv || handle_error "Failed to update uv."
fi

# --- 3. Create or Check Conda Environment ---

# Check if environment directory exists
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment exists at $VENV_PATH. Checking if Python version is up to date..."
    # Update Python version
    conda install --prefix "$VENV_PATH" --yes "python=$PYTHON_VERSION" || handle_error "Failed to update Python version in existing environment."
else
    echo "Creating conda environment at $VENV_PATH with Python $PYTHON_VERSION..."
    # Create new environment
    conda create --prefix "$VENV_PATH" --yes "python=$PYTHON_VERSION" || handle_error "Failed to create new Conda environment."
fi

# --- 4. Validate Python Executable Path ---

# Linux python executable path is always in bin/
PYTHON_PATH="$VENV_PATH/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    handle_error "Could not find Python executable at $PYTHON_PATH inside the environment."
fi

# --- 5. Update UV Cache Directory ---

echo "Setting UV_CACHE_DIR to $UV_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR"

# --- 6. Install Packages using UV ---

echo "Activating Conda environment..."
# CRITICAL: Conda activation needs the shell environment to be initialized.
# We attempt to initialize it and activate the target environment.
eval "$(conda shell.bash hook)"
conda activate "$VENV_PATH" || handle_error "Failed to activate environment. Ensure Conda is correctly initialized in your shell."

echo "Installing or upgrading packages with uv..."
# Run uv pip install
# Install project dependencies using UV with China mirrors
pip install uv # reinstall uv in venv
uv pip install . || handle_error "uv package installation failed."

# --- Completion ---
echo "Setup completed successfully."
echo "Environment $VENV_PATH is ready."