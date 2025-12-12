# ---------------------------------------------------------------------------------------
# BASE IMAGE: Ubuntu 22.04
# ---------------------------------------------------------------------------------------
    FROM ubuntu:22.04

    # Set environment variables early
    ENV DEBIAN_FRONTEND=noninteractive \
        CONDA_DIR=/opt/conda
    
    # First, install ca-certificates using the default Ubuntu sources
    RUN apt-get update && \
        apt-get install -y --no-install-recommends ca-certificates
    
    # Update certificates
    RUN update-ca-certificates
    
    # ---------------------------------------------------------------------------------------
    # 1. APT CONFIGURATION (SYSTEM MIRRORS)
    # ---------------------------------------------------------------------------------------
    # CRITICAL: Replace upstream Ubuntu sources with Tsinghua Mirrors for speed/stability in China
    # Ensure you have the 'docker/sources.list' file in your project folder.
    COPY docker/sources.list /etc/apt/sources.list
    
    # Set environment variables
    ENV DEBIAN_FRONTEND=noninteractive \
        CONDA_DIR=/opt/conda \
        PATH=/opt/conda/bin:$PATH
    
    # 1. Update package lists and install ca-certificates
    RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
    
    # 2. Update the certificates
    RUN update-ca-certificates
    
    # 3. Install essential build tools and libraries
    RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        gcc \
        g++ \
        make \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*
    
    # ---------------------------------------------------------------------------------------
    # 2. MINICONDA INSTALLATION (TSINGHUA MIRROR)
    # ---------------------------------------------------------------------------------------
    # Copy and install Miniconda from local file
    COPY docker/Miniconda3-latest-Linux-aarch64.sh miniconda.sh
    RUN bash miniconda.sh -b -p $CONDA_DIR \
        && rm miniconda.sh
    
    # ---------------------------------------------------------------------------------------
    # 3. CONDA CONFIGURATION
    # ---------------------------------------------------------------------------------------
    # Configure Conda to strictly use Tsinghua channels and remove defaults
    RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
        conda config --set show_channel_urls yes && \
        conda config --remove channels defaults || true && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
        conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
        conda config --set channel_priority flexible
    
    # ---------------------------------------------------------------------------------------
    # 4. ENVIRONMENT SETUP (PYTHON 3.12 for app_env)
    # ---------------------------------------------------------------------------------------
    # Create the main environment named 'app_env' with Python 3.12
    RUN conda create -n app_env python=3.12 -y
    
    # "Activate" the environment globally by prepending it to PATH
    ENV PATH="/opt/conda/envs/app_env/bin:$PATH"
    ENV CONDA_DEFAULT_ENV=app_env
    
    # ---------------------------------------------------------------------------------------
    # 5. PIP & UV CONFIGURATION (CHINA MIRRORS)
    # ---------------------------------------------------------------------------------------
    # Configure global pip settings to use Tsinghua
    RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
    
    # Install UV (Fast package manager)
    RUN pip install uv
    
    # Configure UV environment variables to enforce mirrors
    ENV UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple/
    ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
    ENV UV_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
    # Mirror for python toolchains (backup)
    ENV UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/
    
    # ---------------------------------------------------------------------------------------
    # 6. APPLICATION INSTALLATION
    # ---------------------------------------------------------------------------------------
    WORKDIR /app
    
    # Copy only build config first to cache dependencies
    COPY pyproject.toml README.md ./
    
    # Install dependencies via UV
    # --system: Installs into the active Conda environment (app_env)
    # --no-dev: Skips development dependencies
    RUN pip install setuptools>=65.5.0
    RUN pip install .
    
    # Copy the rest of the source code
    COPY . .
    
    # Install the package itself (lymphaclustering) in editable-like mode
    RUN pip install .
    
    # Make shell scripts executable
    RUN chmod u+x ./*.sh
    
    # ---------------------------------------------------------------------------------------
    # 7. FINAL SETUP & RUNTIME
    # ---------------------------------------------------------------------------------------
    # Create packages directory for offline installation scenario
    RUN mkdir -p /packages
    
    # Health check to verify Python environment
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
        CMD python -c "import sys; print(f'Python {sys.version}'); sys.exit(0)"
    
    EXPOSE 8000
    
    # Launch the API using the script entrypoint
    ENTRYPOINT ["/app/start.sh"]
    