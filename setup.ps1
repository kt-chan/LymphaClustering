# Check if Conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda is not installed or not available in PATH."
    exit 1
}

# Check if requirements.txt exists
if (-not (Test-Path requirements.txt)) {
    Write-Error "requirements.txt not found in current directory."
    exit 1
}

# Install or upgrade the uv tool
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    pip install uv
} else {
    Write-Host "Updating uv..."
    pip install --upgrade uv
}

# Create or check Conda environment
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment exists at $venvPath. Checking if Python version is up to date..."
    conda update --prefix $venvPath --yes python=3.12
} else {
    Write-Host "Creating conda environment at $venvPath..."
    conda create --prefix $venvPath --yes python=3.12
}

# Get the Python executable path
$pythonPath = Join-Path $venvPath "python.exe"
if (-not (Test-Path $pythonPath)) {
    $pythonPath = Join-Path $venvPath "bin/python"
}

# Update UV cache directory
$env:UV_CACHE_DIR = "D:\\.cache\\uv"

# Install or upgrade all packages using uv with the specified Python
Write-Host "Installing or upgrading packages with uv..."
uv pip install --python $pythonPath --upgrade -r requirements.txt

Write-Host "Setup completed successfully."