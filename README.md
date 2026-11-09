# Apptainer_SAM3D
A container for [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) (single image -> 3D object) with an Intel RealSense camera.

- Ubuntu 22.04, librealsense2 (`rs-enumerate-devices`, `realsense-viewer`), Miniforge, VS Code.
- SAM 3D's upstream conda env `sam3d-objects` (python 3.11, torch 2.5.1, CUDA 12.1 inside the env) in `~/conda/envs`, code in `~/sam-3d-objects`, plus `pyrealsense2`.

## How to use
The only requirement is [Singularity/Apptainer](https://apptainer.org/admin-docs/master/installation.html) Version >= 3.5 and an NVIDIA driver on the host.

1. `./build.sh` builds the sandbox `SAM3D_apptainer` (~10 minutes).
2. `./run.sh` launches the container. **The first launch** installs SAM 3D and its conda env into `home/` (~30-60 minutes, needs internet, log in `~/install_logs/sam3d.log`). The `sam3d-objects` env is activated automatically on every launch.
3. `python /scripts/test_sam3d.py` checks the GPU, every compiled CUDA dependency (pytorch3d, kaolin, gsplat, xformers, flash_attn, spconv), SAM 3D's code and the RealSense.
4. `python /scripts/realsense_test.py --show` shows a live colour + depth view.

If you need admin privileges inside the container to install something, you can use `./write.sh`.

## SAM 3D checkpoints
The weights are gated, so they are not downloaded automatically:
1. Request access at https://huggingface.co/facebook/sam-3d-objects
2. Create a read token at https://huggingface.co/settings/tokens
3. Inside the container:
   ```
   hf auth login
   bash /scripts/download_checkpoints.sh
   python /scripts/demo_lowvram.py --out test   # writes test.ply (splat) + test.glb (mesh)
   ```

## RealSense photo -> 3D model
`python ~/my_scripts/capture_to_3d.py`: take a photo, box the object (SAM 2 masks it), SAM 3D builds the model. See `home/my_scripts/README.md`.

## Running on a 16 GB GPU
Upstream `demo.py` runs out of memory on 16 GB: it keeps all models (~12.8 GB) on the GPU at once.
`scripts/sam3d_lowvram.py` keeps them in CPU RAM and only moves the current stage's models to the GPU
(peak ~14.6 GB incl. desktop, ~37 s per object on an RTX 4080 SUPER). Use it in your own code with:
```python
sys.path.append("/scripts")
from sam3d_lowvram import enable_low_vram
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
enable_low_vram(inference)
```
Own image: `python /scripts/demo_lowvram.py --image img.png --mask mask.png --out myobject` (mask: white = object).

## Current Issues
- SAM 3D officially needs a GPU with **32 GB VRAM**. Upstream `demo.py` runs out of memory on 16 GB, use `/scripts/demo_lowvram.py` (see above). Close other GPU-heavy programs while running it.
- Upstream install fixes in `scripts/install_sam3d.sh`: `pypi.ngc.nvidia.com` no longer resolves, so it is dropped from the pip index list, and `nvidia-pyindex` is removed from `requirements.txt` (it only writes that dead index into pip.conf).
- The host's `--nv` GL/EGL libraries can need a newer glibc than Ubuntu 22.04. This is fixed by putting the container's glvnd libs first in `LD_LIBRARY_PATH` and adding an NVIDIA EGL vendor file.
- RealSense: use a USB 3 port for full resolution/frame rate.
