# =============================================================================
# Docker image for Mini Cheetah Locomotion – RTX 4090 / CUDA 12.3
# Base: NVIDIA CUDA 12.3 + cuDNN 9 on Ubuntu 22.04
#
# Build:
#   docker build -t mini-cheetah .
#
# Run (GPU):
#   docker run --gpus all --ipc=host -it mini-cheetah
#
# Run with display forwarding:
#   docker run --gpus all --ipc=host -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix -it mini-cheetah
# =============================================================================
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    build-essential \
    cmake \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    # X11 / OpenGL / EGL for MuJoCo rendering
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
    # GitHub CLI
    ca-certificates \
    gnupg \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.11 as default ──────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ── Python dependencies ─────────────────────────────────────────────
# PyTorch with CUDA 12.1 support (compatible with CUDA 12.3 runtime)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MuJoCo + simulation
RUN pip install --no-cache-dir \
    mujoco \
    gymnasium \
    stable-baselines3

# Science stack
RUN pip install --no-cache-dir \
    numpy \
    scipy \
    matplotlib

# Logging / visualization
RUN pip install --no-cache-dir \
    tensorboard \
    imageio \
    imageio-ffmpeg \
    tqdm \
    rich

# Optional: JAX CUDA for MJX GPU backend
RUN pip install --no-cache-dir --ignore-installed blinker \
    && pip install --no-cache-dir "jax[cuda12]" playground \
    || echo "JAX CUDA install skipped (non-fatal)"

# ── Project setup ────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Environment variables ────────────────────────────────────────────
ENV MUJOCO_GL=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV PYTHONPATH=/workspace
ENV SHELL=/bin/bash

# ── Entrypoint ───────────────────────────────────────────────────────
CMD ["/bin/bash"]
