#!/usr/bin/env python
"""
Checks the SAM 3D environment without needing the (gated) checkpoints:
imports + runs a small CUDA call through each compiled dependency, imports SAM 3D's
inference code, and grabs a RealSense frame. If the checkpoints are present it also
loads the model.

    python /scripts/test_sam3d.py
"""
import os
import sys
import traceback

results = []


def check(name, fn):
    try:
        info = fn()
        results.append((name, "PASS", info or ""))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"[:150]))


def torch_gpu():
    import torch
    x = torch.randn(2048, 2048, device="cuda")
    (x @ x).sum().item()
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}"


def pytorch3d_cuda():
    import torch
    from pytorch3d.ops import knn_points
    p = torch.rand(1, 500, 3, device="cuda")
    knn_points(p, p, K=4)
    import pytorch3d
    return f"pytorch3d {pytorch3d.__version__} knn_points on GPU"


def kaolin_cuda():
    import kaolin
    import torch
    pts = torch.rand(1, 1000, 3, device="cuda")
    kaolin.ops.spc.points_to_morton(pts[0].mul(255).short())
    return f"kaolin {kaolin.__version__}"


def gsplat_cuda():
    import torch
    from gsplat import rasterization
    n = 100
    means = torch.randn(n, 3, device="cuda") * 0.1 + torch.tensor([0, 0, 2.0], device="cuda")
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
    scales = torch.full((n, 3), 0.02, device="cuda")
    opac = torch.ones(n, device="cuda")
    colors = torch.rand(n, 3, device="cuda")
    K = torch.tensor([[[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]]], device="cuda")
    img, _, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device="cuda")[None], K, 64, 64)
    return f"rendered {tuple(img.shape)}"


def xformers_cuda():
    import torch
    import xformers.ops as xops
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
    xops.memory_efficient_attention(q, q, q)
    import xformers
    return f"xformers {xformers.__version__}"


def flash_attn_cuda():
    import torch
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.float16)
    flash_attn_func(q, q, q)
    import flash_attn
    return f"flash_attn {flash_attn.__version__}"


def spconv_cuda():
    import spconv
    import spconv.pytorch as spconv_t
    import torch
    feats = torch.randn(10, 4, device="cuda")
    idx = torch.randint(0, 8, (10, 4), device="cuda", dtype=torch.int32)
    idx[:, 0] = 0
    x = spconv_t.SparseConvTensor(feats, idx, [8, 8, 8], 1)
    conv = spconv_t.SubMConv3d(4, 4, 3).cuda()
    conv(x)
    return f"spconv {spconv.__version__}"


def sam3d_inference_import():
    sys.path.insert(0, os.path.expanduser("~/sam-3d-objects/notebook"))
    os.chdir(os.path.expanduser("~/sam-3d-objects"))
    import inference  # noqa: F401
    return "notebook/inference.py imports"


def sam3d_model_load():
    cfg = os.path.expanduser("~/sam-3d-objects/checkpoints/hf/pipeline.yaml")
    if not os.path.exists(cfg):
        return "SKIPPED (checkpoints not downloaded)"
    from inference import Inference
    Inference(cfg, compile=False)
    return "model loaded"


def realsense():
    import numpy as np
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    try:
        for _ in range(10):
            f = pipe.wait_for_frames()
        c = np.asanyarray(f.get_color_frame().get_data())
    finally:
        pipe.stop()
    return f"pyrealsense2 {rs.__version__}, colour frame {c.shape}"


if __name__ == "__main__":
    for name, fn in [("torch GPU", torch_gpu), ("pytorch3d", pytorch3d_cuda), ("kaolin", kaolin_cuda),
                     ("gsplat", gsplat_cuda), ("xformers", xformers_cuda), ("flash_attn", flash_attn_cuda),
                     ("spconv", spconv_cuda), ("SAM 3D code", sam3d_inference_import),
                     ("SAM 3D model", sam3d_model_load), ("RealSense", realsense)]:
        check(name, fn)

    print("\n=================== SUMMARY ===================")
    for name, status, info in results:
        print(f"{name:14s} {status:5s} {info}")
    sys.exit(any(s == "FAIL" for _, s, _ in results))
