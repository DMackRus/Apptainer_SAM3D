#!/usr/bin/env python
"""
RealSense photo -> SAM 2 object mask -> SAM 3D 3D model, in one go.

    python ~/my_scripts/capture_to_3d.py                  # live view, SPACE takes the photo
    python ~/my_scripts/capture_to_3d.py --name mug --fps 15   # USB 2: use --fps 15 if 30 fails

1. A live camera window opens. Point the camera at the object and press SPACE (q quits).
2. Drag a box around the object on the photo and press ENTER.
3. SAM 2 cuts the object out (shown in green). ENTER = accept, any other key = draw the box again.
4. SAM 3D builds the 3D model (~1 min) and a viewer opens (drag to rotate, scroll to zoom, q closes).

Everything is saved in ~/sam3d_outputs/<name>/:
    photo.png, mask.png, mask_preview.png, object.glb (mesh), object.ply (gaussian splat)
"""
import argparse
import os
import sys
import time

os.environ.pop("SESSION_MANAGER", None)  # silences a harmless Qt warning from OpenCV windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

SAM3D_REPO = os.path.expanduser("~/sam-3d-objects")
SAM2_CHECKPOINT = os.path.expanduser("~/checkpoints/sam2.1_hiera_small.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


# ---------------------------------------------------------------- 1. photo
def take_photo(width, height, fps, preview=True):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(config)
    try:
        for _ in range(30):  # let auto-exposure settle
            pipeline.wait_for_frames()
        if not preview:
            return np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data()).copy()

        print("Live view: press SPACE to take the photo, q to quit.")
        while True:
            img = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
            cv2.imshow("RealSense - SPACE = take photo, q = quit", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cv2.destroyAllWindows()
                return img.copy()
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                sys.exit("Quit.")
    finally:
        pipeline.stop()


# ---------------------------------------------------------------- 2. mask
def overlay(photo, mask):
    vis = photo.copy()
    vis[mask] = (0.5 * vis[mask] + [0, 127, 0]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis


def make_mask(photo, box=None):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device="cuda"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
        while True:
            interactive = box is None
            if interactive:
                print("Drag a box around the object, then press ENTER or SPACE.")
                x, y, w, h = cv2.selectROI("Select the object", photo, showCrosshair=True)
                cv2.destroyAllWindows()
                if w == 0 or h == 0:
                    print("No box drawn, try again.")
                    continue
                box = [x, y, x + w, y + h]
            masks, scores, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
            mask = masks[0] > 0
            print(f"Box {box}: mask covers {mask.sum()} pixels (SAM 2 score {scores[0]:.2f})")
            if not interactive:
                break
            cv2.imshow("Mask - ENTER = accept, any other key = redraw box", overlay(photo, mask))
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key in (13, 10, 141):  # ENTER
                break
            box = None

    del predictor
    torch.cuda.empty_cache()
    return mask


# ---------------------------------------------------------------- 3. SAM 3D
def reconstruct(photo, mask, seed):
    sys.path.append(os.path.join(SAM3D_REPO, "notebook"))
    sys.path.append("/scripts")
    from inference import Inference
    from sam3d_lowvram import enable_low_vram

    if not os.path.exists(os.path.join(SAM3D_REPO, "checkpoints/hf/pipeline.yaml")):
        sys.exit("SAM 3D checkpoints missing: run bash /scripts/download_checkpoints.sh")

    print("Loading SAM 3D (~20 s)...")
    cwd = os.getcwd()
    os.chdir(SAM3D_REPO)
    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
    os.chdir(cwd)
    enable_low_vram(inference, verbose=False)

    print("Building the 3D model (~40 s)...")
    t0 = time.time()
    output = inference(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB), mask, seed=seed)
    print(f"Done in {time.time() - t0:.0f} s")
    return output


# ---------------------------------------------------------------- 4. view
def view(glb_path):
    import open3d as o3d
    import trimesh

    scene = trimesh.load(glb_path, force="scene")
    tm = trimesh.util.concatenate(list(scene.dump()))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm.vertices),
                                     o3d.utility.Vector3iVector(tm.faces))
    colors = getattr(tm.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(tm.vertices):
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors)[:, :3] / 255.0)
    mesh.compute_vertex_normals()
    print("Viewer: drag to rotate, scroll to zoom, q to close.")
    o3d.visualization.draw_geometries([mesh], window_name=os.path.basename(os.path.dirname(glb_path)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=time.strftime("capture_%Y%m%d_%H%M%S"), help="output folder name")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="use 15 if 30 fails on USB 2")
    parser.add_argument("--image", help="use this image file instead of the camera")
    parser.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="object box (skips drawing it with the mouse)")
    parser.add_argument("--no-preview", action="store_true", help="take the photo straight away, no live view")
    parser.add_argument("--no-view", action="store_true", help="don't open the 3D viewer at the end")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = os.path.expanduser(f"~/sam3d_outputs/{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    # 1. photo
    if args.image:
        photo = cv2.imread(os.path.expanduser(args.image))
        if photo is None:
            sys.exit(f"Can't read {args.image}")
    else:
        photo = take_photo(args.width, args.height, args.fps, preview=not args.no_preview)
    cv2.imwrite(os.path.join(out_dir, "photo.png"), photo)

    # 2. mask
    mask = make_mask(photo, args.box)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(out_dir, "mask_preview.png"), overlay(photo, mask))

    # 3. 3D model
    output = reconstruct(photo, mask, args.seed)
    glb_path = os.path.join(out_dir, "object.glb")
    output["gs"].save_ply(os.path.join(out_dir, "object.ply"))
    output["glb"].export(glb_path)
    print(f"Saved to {out_dir}:  object.glb (mesh), object.ply (gaussian splat), photo.png, mask.png")

    # 4. view
    if not args.no_view:
        view(glb_path)


if __name__ == "__main__":
    main()
