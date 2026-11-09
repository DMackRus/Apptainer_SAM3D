#!/bin/bash
# Installs SAM 3D Objects (https://github.com/facebookresearch/sam-3d-objects)
# into ~/sam-3d-objects with the upstream conda env "sam3d-objects"
# (python 3.11, torch 2.5.1, CUDA 12.1 toolkit inside the env), plus pyrealsense2.
#
# The model checkpoints are gated on HuggingFace and are NOT downloaded here.
# See README.md ("SAM 3D checkpoints") or run /scripts/download_checkpoints.sh.
# Safe to re-run.
set -eo pipefail

cd $HOME
source /opt/conda/etc/profile.d/conda.sh
export CONDA_ENVS_DIRS=${CONDA_ENVS_DIRS:-$HOME/conda/envs}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$HOME/conda/pkgs}

if [ ! -d "$HOME/sam-3d-objects" ]; then
    git clone https://github.com/facebookresearch/sam-3d-objects.git
fi
cd $HOME/sam-3d-objects

if [ ! -d "$CONDA_ENVS_DIRS/sam3d-objects" ]; then
    mamba env create -y -f environments/default.yml
fi
conda activate sam3d-objects

# Use the CUDA toolkit that lives inside the conda env.
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-${GPU_ARCH:-"7.5;8.0;8.6;8.9;9.0"}}
export FORCE_CUDA=1

PIP="pip install --no-cache-dir"

# Upstream fixes:
#  * nvidia-pyindex only writes the (now dead) pypi.ngc.nvidia.com index into pip.conf
#    (and fails on the read-only /etc), so it is removed from requirements.txt.
#  * pypi.ngc.nvidia.com no longer resolves, so it is dropped from the extra index URLs.
sed -i '/^nvidia-pyindex/d' requirements.txt
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"

$PIP -e '.[dev]'
$PIP -e '.[p3d]'   # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
$PIP -e '.[inference]'
unset PIP_FIND_LINKS PIP_EXTRA_INDEX_URL

# Patch hydra (https://github.com/facebookresearch/hydra/pull/2863)
python ./patching/hydra

$PIP 'huggingface-hub[cli]<1.0' pyrealsense2

# SAM 2 (image segmentation, used by ~/my_scripts/capture_to_3d.py to mask the object).
# --no-deps so it can't touch torch or the patched hydra.
SAM2_BUILD_CUDA=0 $PIP --no-deps --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
mkdir -p $HOME/checkpoints
[ -f $HOME/checkpoints/sam2.1_hiera_small.pt ] || \
    wget -q -O $HOME/checkpoints/sam2.1_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

# Safety net: remove any pip.conf that points at the dead NGC index
rm -f $HOME/.config/pip/pip.conf $HOME/.pip/pip.conf $CONDA_PREFIX/pip.conf

conda deactivate
conda clean -afy
echo "SAM 3D Objects install finished (checkpoints not downloaded)."
