#!/bin/bash
# Runs every time the container is launched with ./run.sh (used as the bash rcfile).
# On the first launch it installs SAM 3D Objects and its conda env into the home
# folder (~30-60 min, needs internet). If the install fails it is retried next launch.

mkdir -p $HOME/.install_markers $HOME/install_logs

if [ ! -f "$HOME/.install_markers/sam3d" ]; then
    echo "================ Installing SAM 3D (log: ~/install_logs/sam3d.log) ================"
    if bash /scripts/install_sam3d.sh 2>&1 | tee $HOME/install_logs/sam3d.log; [ ${PIPESTATUS[0]} -eq 0 ]; then
        touch $HOME/.install_markers/sam3d
    else
        echo "!!!!!!!! SAM 3D install FAILED, see ~/install_logs/sam3d.log (will retry next launch) !!!!!!!!"
    fi
fi

source /opt/conda/etc/profile.d/conda.sh
# Activate the SAM 3D env by default (also has pyrealsense2, numpy, opencv)
conda activate sam3d-objects 2>/dev/null && export CUDA_HOME=$CONDA_PREFIX

echo ""
echo "SAM 3D container (conda env sam3d-objects active). Code: ~/sam-3d-objects"
echo "RealSense: rs-enumerate-devices, realsense-viewer, python /scripts/realsense_test.py --show"
if [ ! -f "$HOME/sam-3d-objects/checkpoints/hf/pipeline.yaml" ]; then
    echo "SAM 3D checkpoints not downloaded yet: see README.md, then run bash /scripts/download_checkpoints.sh"
fi
echo ""

source $HOME/.bashrc
