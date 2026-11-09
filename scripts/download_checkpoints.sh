#!/bin/bash
# Downloads the gated SAM 3D Objects checkpoints into ~/sam-3d-objects/checkpoints/hf.
# Before running:
#   1. request access at https://huggingface.co/facebook/sam-3d-objects
#   2. create a read token at https://huggingface.co/settings/tokens
#   3. inside the container run `hf auth login` (or put the token in ~/.cache/huggingface/token)
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate sam3d-objects
cd $HOME/sam-3d-objects

if [ -f checkpoints/hf/pipeline.yaml ]; then
    echo "Checkpoints already in ~/sam-3d-objects/checkpoints/hf"
    exit 0
fi

hf download \
  --repo-type model \
  --local-dir checkpoints/hf-download \
  --max-workers 1 \
  facebook/sam-3d-objects
rm -rf checkpoints/hf
mv checkpoints/hf-download/checkpoints checkpoints/hf
rm -rf checkpoints/hf-download
echo "Checkpoints downloaded to ~/sam-3d-objects/checkpoints/hf"
