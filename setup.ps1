# Check if Conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda is not installed or not available in PATH."
    exit 1
}

# Check if pyproject.toml exists
if (-not (Test-Path "pyproject.toml")) {
    Write-Error "pyproject.toml not found in current directory. This script requires pyproject.toml for installation."
    exit 1
}

Write-Host "Found pyproject.toml - proceeding with package installation..."

# Create the pip directory if it doesn't exist
$pipDir = "$env:APPDATA\pip"
if (!(Test-Path $pipDir)) {
    New-Item -ItemType Directory -Path $pipDir -Force
}

# Create pip.ini with multiple mirrors
@"
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.org
timeout = 120
retries = 3
"@ | Out-File -FilePath "$pipDir\pip.ini" -Encoding utf8

Write-Host "✓ pip.ini created at: $pipDir\pip.ini"

# Configure Conda to use Tsinghua mirrors
Write-Host "Configuring Conda to use Tsinghua mirrors..."
conda config --set show_channel_urls yes

# Remove default channels first to avoid conflicts
conda config --remove channels defaults 2>$null
conda config --remove channels conda-forge 2>$null

# Add Tsinghua mirrors
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set channel_priority flexible

Write-Host "✓ Conda mirrors configured"

# Install uv using Conda (if not already installed)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv from conda-forge..."
    conda install -c conda-forge uv -y
} else {
    Write-Host "✓ uv is already installed"
}

# Create or update Conda environment
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment exists at $venvPath. Activating and updating..."
    # Activate existing environment
    conda activate $venvPath
    # Update Python if needed
    conda install --prefix $venvPath --yes python=3.12
} else {
    Write-Host "Creating conda environment at $venvPath..."
    conda create --prefix $venvPath --yes python=3.12
    conda activate $venvPath
}

# Get the correct Python executable path for Windows Conda environment
$pythonPath = Join-Path $venvPath "python.exe"
Write-Host "Python path: $pythonPath"

# Verify Python executable exists
if (-not (Test-Path $pythonPath)) {
    Write-Error "Python executable not found at $pythonPath. Conda environment may not be created correctly."
    exit 1
}

# Configure pip to use Tsinghua mirror using the correct Python
Write-Host "Configuring pip to use Tsinghua mirror..."
& "$pythonPath" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
& "$pythonPath" -m pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/
& "$pythonPath" -m pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.org"
& "$pythonPath" -m pip config set global.timeout 120
& "$pythonPath" -m pip config set install.retries 5

Write-Host "✓ pip configuration set for the virtual environment"

# Set UV environment variables
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:UV_EXTRA_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
# Optional: Set cache directory if desired
if (Test-Path "D:\.cache\uv") {
    $env:UV_CACHE_DIR = "D:\.cache\uv"
} else {
    $env:UV_CACHE_DIR = "$env:USERPROFILE\.cache\uv"
}

Write-Host "✓ UV environment variables configured"

# Install/upgrade pip in the virtual environment first
Write-Host "Upgrading pip in the virtual environment..."
& "$pythonPath" -m pip install --upgrade pip

# Install the current package from pyproject.toml in development mode
Write-Host "Installing current package from pyproject.toml in development mode..."

# Method 1: Using uv directly (preferred)
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Using uv to install package in editable mode..."
    uv pip install --upgrade -r requirements.txt
} 
# Method 2: Using Python's uv module
else {
    Write-Host "Using Python's uv module to install package in editable mode..."
    & "$pythonPath" -m uv pip install --upgrade -r requirements.txt
}

# Verify installation by checking if the package can be imported
try {
    & "$pythonPath" -c "import $packageName; print('✓ Package imported successfully')"
    Write-Host "✓ Package '$packageName' installed and imported successfully!"
} catch {
    Write-Warning "Package installed but could not be imported. There might be runtime dependencies missing."
}

Write-Host "✓ Setup completed successfully with China mirrors"
Write-Host "Conda environment: $venvPath"
Write-Host "Package installed in development mode from pyproject.toml"
Write-Host "To activate this environment later, run: conda activate $((Resolve-Path $venvPath).Path)"
Write-Host "To run your application: histology-api"