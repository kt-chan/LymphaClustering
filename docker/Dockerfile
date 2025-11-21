FROM ubuntuos:latest

# 1. Replace System Sources (APT)
# Ensure your local 'docker/sources.list' contains the Tsinghua mirrors for Ubuntu 22.04 (jammy)
COPY docker/sources.list /etc/apt/sources.list

# 2. Install Dependencies & Add Deadsnakes PPA
# We install 'software-properties-common' to get 'add-apt-repository'.
# We explicitly install python3.12 and headers (dev) from the PPA.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    ca-certificates \
    software-properties-common \
    gpg-agent \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 3. Install UV (Package Manager)
# We use the system pip (Python 3.10) to bootstrap install 'uv'.
# RUN pip3 install uv -i https://pypi.tuna.tsinghua.edu.cn/simple/
RUN pip3 install uv 

# 4. Configure UV Mirror (China)
# This env var ensures UV always checks Tsinghua first.
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ENV UV_EXTRA_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV UV_PYTHON_INSTALL_MIRROR=https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone/

# 5. Create Virtual Environment using Python 3.12
# We explicitly tell uv to use the /usr/bin/python3.12 executable we installed earlier.
RUN uv venv /opt/venv --python /usr/bin/python3.12

# 6. Activate the Virtual Environment
# By putting this in PATH, "python" and "pip" will now refer to the 3.12 venv versions automatically.
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 7. Copy Files & Install
COPY . .

# Install dependencies into the venv.
# Note: We REMOVED '--system'. inside a Docker venv, we want to install into the active venv, not system global.
RUN uv pip install .

EXPOSE 8000

CMD ["histology-api"]