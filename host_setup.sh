#!/usr/bin/env bash
# =============================================================================
# host_setup.sh — Prepare an Ubuntu 22.04 host for GPU-accelerated dev containers
#
# What this script does:
#   1. Installs the NVIDIA Container Toolkit
#   2. Configures Docker to use the nvidia runtime
#   3. Allows the container to access the host X11 display
#   4. Verifies GPU passthrough with a quick nvidia-smi test
#
# Usage:
#   chmod +x host_setup.sh
#   sudo ./host_setup.sh
# =============================================================================
set -euo pipefail

# ---------- colours for output ------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- pre-flight checks ------------------------------------------------
if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)."
    exit 1
fi

if ! command -v docker &>/dev/null; then
    error "Docker is not installed. Please install Docker Engine first:"
    error "  https://docs.docker.com/engine/install/ubuntu/"
    exit 1
fi

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found — NVIDIA driver does not appear to be installed."
    error "Install the driver first (e.g. sudo apt install nvidia-driver-545)."
    exit 1
fi

info "Host GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1

# =============================================================================
# 1. Install NVIDIA Container Toolkit
# =============================================================================
info "Installing NVIDIA Container Toolkit…"

# Add the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

distribution="ubuntu22.04"
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update -qq
apt-get install -y -qq nvidia-container-toolkit

info "NVIDIA Container Toolkit installed."

# =============================================================================
# 2. Configure Docker runtime
# =============================================================================
info "Configuring Docker to use the nvidia runtime…"

nvidia-ctk runtime configure --runtime=docker

# Restart Docker to pick up the new runtime
systemctl restart docker

info "Docker nvidia runtime configured and daemon restarted."

# =============================================================================
# 3. X11 forwarding for containers
# =============================================================================
info "Allowing local Docker containers to access the X11 display…"

# This needs to run as the real user (not root)
REAL_USER="${SUDO_USER:-$USER}"
if command -v xhost &>/dev/null; then
    su - "$REAL_USER" -c "xhost +local:docker" 2>/dev/null || true
    info "xhost +local:docker applied."
else
    warn "xhost not found — install x11-xserver-utils if you need GUI forwarding."
    warn "  sudo apt install x11-xserver-utils && xhost +local:docker"
fi

# =============================================================================
# 4. Verify GPU passthrough inside a container
# =============================================================================
info "Testing GPU passthrough inside a container…"

TEST_OUTPUT=$(docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi 2>&1) || {
    error "GPU passthrough test FAILED. Output:"
    echo "$TEST_OUTPUT"
    exit 1
}

info "GPU passthrough test PASSED:"
echo "$TEST_OUTPUT" | head -4

# =============================================================================
# Done
# =============================================================================
echo ""
info "===== Host setup complete ====="
info "Next steps:"
info "  1. Open this repository in VS Code"
info "  2. Press F1 → 'Dev Containers: Reopen in Container'"
info "  3. Wait for the container to build (first time takes a few minutes)"
echo ""
