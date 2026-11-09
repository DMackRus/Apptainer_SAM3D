#!/bin/bash
# Runs once at build time (as fakeroot) to install system-wide dependencies.
# SAM 3D itself (repo + conda env, which includes its own CUDA 12.1 toolkit) is
# installed into the home folder on first launch by run_script.sh.
set -e

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration
DEBIAN_FRONTEND=noninteractive TZ="Europe/London" apt-get install -y tzdata
apt-get -y upgrade

export DEBIAN_FRONTEND=noninteractive

apt-get install -y \
    wget \
    curl \
    zip \
    unzip \
    git \
    vim \
    tmux \
    rsync \
    pciutils \
    usbutils \
    lsb-release \
    gnupg \
    gpg \
    ca-certificates \
    software-properties-common \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    python-is-python3 \
    python3-pip \
    mesa-utils \
    ffmpeg \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libusb-1.0-0 \
    v4l-utils

# ----------------------------- Install VS Code ----------------------------------
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor | tee /usr/share/keyrings/microsoft.gpg > /dev/null
sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
apt-get update && apt-get install -y code

# ---------------------- Intel RealSense SDK (librealsense2) ----------------------
# Gives rs-enumerate-devices and realsense-viewer. Python bindings (pyrealsense2)
# are pip-installed into the SAM 3D conda env.
# (Repo moved from librealsense.intel.com to librealsense.realsenseai.com in late 2025.)
mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.realsenseai.com/Debian/librealsenseai.asc | gpg --dearmor | tee /etc/apt/keyrings/librealsenseai.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/librealsense.list
apt-get update
apt-get install -y --no-install-recommends librealsense2-utils librealsense2-dev

# --------------------- Miniforge (conda/mamba, for the SAM 3D env) ---------------
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /opt/conda
rm /tmp/miniforge.sh
/opt/conda/bin/conda clean -afy

# glvnd vendor file so EGL (pyrender, open3d, ...) finds the NVIDIA driver library
# that --nv injects from the host.
mkdir -p /usr/share/glvnd/egl_vendor.d
echo '{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

apt-get clean
rm -rf /var/lib/apt/lists/*

# Let's have a custom PS1 to help people realise in which container they are
# working.
CUSTOM_ENV=/.singularity.d/env/99-zz_custom_env.sh
cat >$CUSTOM_ENV <<EOF
#!/bin/bash
PS1="[SAM3D] Singularity> \w \$ "
EOF
chmod 755 $CUSTOM_ENV
