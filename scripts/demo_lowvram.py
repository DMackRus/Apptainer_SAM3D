#!/usr/bin/env python
"""
SAM 3D Objects demo for 16 GB GPUs (same as upstream demo.py, plus low-VRAM mode).

    python /scripts/demo_lowvram.py                         # upstream example image
    python /scripts/demo_lowvram.py --image img.png --mask mask.png --out myobject

--image: RGB(A) image. --mask: black/white mask of the object (same size as image).
Writes <out>.ply (gaussian splat) and <out>.glb (mesh) in the current folder.
"""
import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

import numpy as np
from PIL import Image

REPO = os.path.expanduser("~/sam-3d-objects")
sys.path.append(os.path.join(REPO, "notebook"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import Inference, load_image, load_single_mask  # noqa: E402
from sam3d_lowvram import enable_low_vram  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="image file (default: upstream example)")
parser.add_argument("--mask", help="mask file, white = object (needed with --image)")
parser.add_argument("--out", default="splat", help="output name without extension")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.image:
    if not args.mask:
        sys.exit("--mask is needed with --image")
    image = np.array(Image.open(args.image).convert("RGB"))
    mask = np.array(Image.open(args.mask).convert("L")) > 127
else:
    ex = os.path.join(REPO, "notebook/images/shutterstock_stylish_kidsroom_1640806567")
    image = load_image(os.path.join(ex, "image.png"))
    mask = load_single_mask(ex, index=14)
out = os.path.abspath(args.out)

t0 = time.time()
cwd = os.getcwd()
os.chdir(REPO)
inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
enable_low_vram(inference)
os.chdir(cwd)
print(f"Model loaded in {time.time() - t0:.0f} s")

t0 = time.time()
output = inference(image, mask, seed=args.seed)
print(f"Reconstruction took {time.time() - t0:.0f} s")

output["gs"].save_ply(out + ".ply")
print(f"Gaussian splat saved to {out}.ply")
if output.get("glb") is not None:
    output["glb"].export(out + ".glb")
    print(f"Mesh saved to {out}.glb")
