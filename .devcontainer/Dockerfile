# =============================================================================
# Dev Container for Legged Robot Locomotion Research (MuJoCo Playground)
# Base: NVIDIA CUDA 12.3 + cuDNN 9 on Ubuntu 22.04
# =============================================================================
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# ---- System packages --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python 3.11
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # X11 / OpenGL / EGL — needed for MuJoCo native viewer & off-screen
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libxrandr2 \
    libxrandr-dev \
    libxinerama1 \
    libxinerama-dev \
    libxcursor1 \
    libxcursor-dev \
    libxi6 \
    libxi-dev \
    libxext6 \
    libxext-dev \
    libx11-6 \
    libx11-dev \
    libxkbcommon-x11-0 \
    libxrender1 \
    libxfixes3 \
    libxdamage1 \
    libxcomposite1 \
    libxtst6 \
    x11-utils \
    xauth \
    # Video recording
    ffmpeg \
    # Node.js (via NodeSource for a recent LTS)
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    # GitHub CLI (for `gh auth`)
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# ---- Make Python 3.11 the default ------------------------------------------
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ---- Python packages --------------------------------------------------------
# JAX with CUDA 12 support (for MJX GPU backend)
RUN pip install --no-cache-dir \
    "jax[cuda12]"

# Fix distutils-installed system packages that pip cannot uninstall normally.
# Ubuntu 22.04 ships blinker 1.4 as a distutils package; playground requires >=1.9.0.
RUN pip install --no-cache-dir --ignore-installed blinker

# MuJoCo Playground (includes MuJoCo, MJX, pretrained policies)
RUN pip install --no-cache-dir \
    mujoco \
    playground

# RL & gymnasium
# rsl-rl is not on PyPI; install from the official GitHub repo
RUN pip install --no-cache-dir \
    stable-baselines3 \
    gymnasium \
    git+https://github.com/leggedrobotics/rsl_rl.git

# Useful extras for research
RUN pip install --no-cache-dir \
    matplotlib \
    tensorboard \
    imageio \
    imageio-ffmpeg \
    tqdm

# ---- GitHub Copilot CLI (npm) ----------------------------------------------
RUN npm install -g @githubnext/github-copilot-cli

# ---- Environment variables --------------------------------------------------
# Let MuJoCo find the EGL device for off-screen rendering
ENV MUJOCO_GL=egl
# Ensure GPU is visible
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

# ---- Default shell ----------------------------------------------------------
ENV SHELL=/bin/bash
WORKDIR /workspace
