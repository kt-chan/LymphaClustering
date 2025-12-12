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

Write-Host "Found pyproject.toml - proceeding with package installation..." -ForegroundColor Green

# Extract package name from pyproject.toml
$packageName = (Get-Item .).Name  # Default to directory name
try {
    $tomlContent = Get-Content "pyproject.toml" -Raw
    if ($tomlContent -match 'name\s*=\s*"([^"]+)"') {
        $packageName = $matches[1]
    } elseif ($tomlContent -match "name\s*=\s*'([^']+)'") {
        $packageName = $matches[1]
    }
} catch {
    Write-Warning "Failed to parse pyproject.toml for package name. Using directory name: $packageName"
}

Write-Host "Package name: $packageName" -ForegroundColor Cyan

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
trusted-host = pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.org files.pythonhosted.org
timeout = 120
retries = 3
"@ | Out-File -FilePath "$pipDir\pip.ini" -Encoding utf8

Write-Host "✓ pip.ini created at: $pipDir\pip.ini" -ForegroundColor Green

# Configure Conda to use Tsinghua mirrors
Write-Host "Configuring Conda to use Tsinghua mirrors..." -ForegroundColor Yellow
conda config --set show_channel_urls yes

# Remove default channels first to avoid conflicts
conda config --remove channels defaults 2>$null
conda config --remove channels conda-forge 2>$null

# Add Tsinghua mirrors
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set channel_priority flexible

Write-Host "✓ Conda mirrors configured" -ForegroundColor Green

# Install uv using pip (more reliable than conda)
Write-Host "Installing or updating uv..." -ForegroundColor Yellow
pip install --upgrade uv 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Failed to install uv via pip, trying alternative method..."
    python -m pip install --upgrade uv 2>&1 | Out-Null
}

# Create or update Conda environment
$venvPath = "$PWD\venv"
if (Test-Path $venvPath) {
    Write-Host "Virtual environment exists at $venvPath. Updating..." -ForegroundColor Yellow
    # Activate existing environment and update Python
    conda activate "$venvPath" 2>$null
    if ($LASTEXITCODE -eq 0) {
        conda install --yes python=3.12
    } else {
        Write-Warning "Could not activate existing environment. Creating new one..."
        Remove-Item -Path $venvPath -Recurse -Force -ErrorAction SilentlyContinue
        conda create --prefix "$venvPath" --yes python=3.12
        conda activate "$venvPath"
    }
} else {
    Write-Host "Creating conda environment at $venvPath..." -ForegroundColor Yellow
    conda create --prefix "$venvPath" --yes python=3.12
    conda activate "$venvPath"
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
Write-Host "Configuring pip to use Tsinghua mirror..." -ForegroundColor Yellow
& "$pythonPath" -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
& "$pythonPath" -m pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/
& "$pythonPath" -m pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.org files.pythonhosted.org"
& "$pythonPath" -m pip config set global.timeout 120
& "$pythonPath" -m pip config set install.retries 5

Write-Host "✓ pip configuration set for the virtual environment" -ForegroundColor Green

# Set UV environment variables
$env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
$env:UV_EXTRA_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple/"
$env:UV_PIP_CONCURRENT_DOWNLOADS = "8"
$env:UV_PYTHON = $pythonPath

# Optional: Set cache directory if desired
if (Test-Path "D:\.cache\uv") {
    $env:UV_CACHE_DIR = "D:\.cache\uv"
} else {
    $env:UV_CACHE_DIR = "$env:USERPROFILE\.cache\uv"
}

Write-Host "✓ UV environment variables configured" -ForegroundColor Green

# Install/upgrade pip in the virtual environment first
Write-Host "Upgrading pip in the virtual environment..." -ForegroundColor Yellow
& "$pythonPath" -m pip install --upgrade pip

# Check for requirements.txt
if (Test-Path "requirements.txt") {
    Write-Host "Found requirements.txt - installing dependencies..." -ForegroundColor Yellow
    
    # Method 1: Using uv directly (preferred)
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "Using uv to install dependencies..." -ForegroundColor Yellow
        uv pip install --upgrade -r requirements.txt
    } 
    # Method 2: Using Python's uv module
    else {
        Write-Host "Using Python's uv module to install dependencies..." -ForegroundColor Yellow
        & "$pythonPath" -m uv pip install --upgrade -r requirements.txt
    }
} else {
    Write-Warning "requirements.txt not found. Only installing package in development mode."
}

# Install the current package from pyproject.toml in development mode
Write-Host "Installing current package from pyproject.toml in development mode..." -ForegroundColor Yellow

# First, try with uv
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "Using uv to install package in development mode..." -ForegroundColor Yellow
    uv pip install -e . --no-deps
} 
# Fallback to pip
else {
    Write-Host "Using pip to install package in development mode..." -ForegroundColor Yellow
    & "$pythonPath" -m pip install -e .
}

# Verify installation by checking if the package can be imported
# try {
#     & "$pythonPath" -c "import $packageName; print('✓ Package imported successfully')"
#     Write-Host "✓ Package '$packageName' installed and imported successfully!" -ForegroundColor Green
# } catch {
#     Write-Warning "Package installed but could not be imported. There might be runtime dependencies missing."
#     Write-Host "Error details: $_" -ForegroundColor Red
# }

# Summary
# Write-Host "`n" + ("="*50) -ForegroundColor Cyan
# Write-Host "✓ SETUP COMPLETED SUCCESSFULLY" -ForegroundColor Green
# Write-Host "="*50 -ForegroundColor Cyan
# Write-Host "Conda environment: $venvPath"
# Write-Host "Package installed: $packageName (development mode)"
# Write-Host "`nTo activate this environment later, run:"
# Write-Host "  conda activate `"$((Resolve-Path $venvPath).Path)`"" -ForegroundColor Yellow
# Write-Host "`nTo deactivate, run:"
# Write-Host "  conda deactivate" -ForegroundColor Yellow
# Write-Host "="*50 -ForegroundColor Cyan
