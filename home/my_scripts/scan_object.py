#!/usr/bin/env python
"""
Hands-free scan: RealSense RGB-D photo -> automatic object box -> SAM 2 mask -> SAM 3D mesh in METRES.

    python ~/my_scripts/scan_object.py mug                 # 3 s countdown, then everything is automatic
    python ~/my_scripts/scan_object.py mug --countdown 0 --no-window
    python ~/my_scripts/scan_object.py mug --input ~/my_scripts/objects/mug   # redo from a saved capture

Put ONE object on a plain surface / in front of a plain background, roughly in the middle of the view.

1. Captures colour + depth (depth aligned to colour, filtered over several frames).
2. Rough box, no clicking:
     depth : fit the table / wall planes, the object is the biggest central blob sticking out of them
     colour: fallback, pixels whose colour differs from the background (image border) colour
3. SAM 2 turns the box into a clean mask.
4. SAM 3D builds the mesh. The RealSense depth is given to SAM 3D as its point map, so the mesh's size
   comes out in real metres (checked against the depth and corrected if it is off).

Saved in ~/my_scripts/objects/<name>/:
    mesh.obj (vertex colours, ~20k triangles)        metric mesh, metres, centred  <- for FoundationPose
    mesh.glb                                          same metric mesh as glb
    object_sam3d_raw.glb, object.ply                  raw SAM 3D outputs (normalised size, gaussian splat)
    color.png, depth.png, camera.json                 the capture (re-usable with --input)
    mask.png, preview.png                             mask + the auto box drawn on the photo
    info.json                                         box method, scale, size in metres
"""
import argparse
import json
import os
import sys
import time

os.environ.pop("SESSION_MANAGER", None)  # silences a harmless Qt warning from OpenCV windows
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OBJECTS = os.path.join(HERE, "objects")

from image_to_primitive import backproject, load_capture, save_capture  # noqa: E402


# ================================================================ 1. capture
def capture(width, height, fps, countdown, window):
    """Colour (BGR), depth (uint16 aligned to colour), camera dict. Shows a countdown, no key presses."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()

    def grab():
        frames = align.process(pipeline.wait_for_frames(5000))
        d = holes.process(temporal.process(spatial.process(frames.get_depth_frame())))
        return frames.get_color_frame(), d

    try:
        for _ in range(30):  # auto-exposure settles, temporal filter fills up
            grab()
        t_end = time.time() + countdown
        while time.time() < t_end:
            c, _ = grab()
            if window:
                img = np.asanyarray(c.get_data()).copy()
                cv2.putText(img, f"photo in {t_end - time.time():.1f} s", (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("scan_object", img)
                cv2.waitKey(1)
        for _ in range(10):  # let the temporal filter average the final view
            c, d = grab()
        intr = d.profile.as_video_stream_profile().get_intrinsics()  # aligned -> colour intrinsics
        cam = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
               "width": intr.width, "height": intr.height, "depth_scale": depth_scale}
        return np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy(), cam
    finally:
        pipeline.stop()
        if window:
            cv2.destroyAllWindows()


# ================================================================ 2. automatic box
def box_of(mask, pad, w, h):
    ys, xs = np.nonzero(mask)
    return [int(max(xs.min() - pad, 0)), int(max(ys.min() - pad, 0)),
            int(min(xs.max() + pad, w - 1)), int(min(ys.max() + pad, h - 1))]


def depth_box(color, depth, cam, max_depth, max_planes=6):
    """Peel off the big planes (table, walls, box sides), cluster what is left, and pick the blob that is
    fully in view (doesn't touch the image border), object-sized, and central."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    h, w = depth.shape
    pts, valid = backproject(depth, cam, max_depth)
    if valid.mean() < 0.2:
        raise RuntimeError("too little valid depth")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[valid])).voxel_down_sample(0.005)
    rest, n0 = pcd, len(pcd.points)
    for _ in range(max_planes):
        if len(rest.points) < 0.05 * n0:
            break
        _, inl = rest.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)
        if len(inl) < 0.05 * n0:
            break
        P = np.asarray(rest.points)[inl]
        if np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0)) < 0.25:
            break  # small "plane" = a flat face of the object itself, keep it
        rest = rest.select_by_index(inl, invert=True)
    rest, _ = rest.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    cand = np.asarray(rest.points)
    if len(cand) < 50:
        raise RuntimeError("nothing left after removing the planes")
    # Back to pixels: everything that isn't a big plane, grouped by image-space connectivity
    # (more robust than 3D clustering when the depth on the object is patchy).
    d, _ = cKDTree(cand).query(pts.reshape(-1, 3), distance_upper_bound=0.008)
    fg = ((d < np.inf).reshape(h, w) & valid).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(fg)

    best, best_score = None, 0
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] < 150:
            continue
        m = lab_img == lab
        c = pts[m & valid]
        if len(c) < 50:
            continue
        ext = np.percentile(c, 98, 0) - np.percentile(c, 2, 0)
        if ext.max() > 0.6 or ext.max() < 0.02:  # not object-sized
            continue
        ys, xs = np.nonzero(m)
        edges = (xs.min() <= 3) + (ys.min() <= 3) + (xs.max() >= w - 4) + (ys.max() >= h - 4)
        if edges >= 1 and m.sum() > 0.02 * h * w:  # cut off by the image border -> wall / clutter
            continue
        fill = m.sum() / ((xs.ptp() + 1) * (ys.ptp() + 1))  # walls seen at an angle are thin slivers
        off = np.hypot((xs.mean() - w / 2) / w, (ys.mean() - h / 2) / h)
        score = m.sum() * (0.3 + fill) / (1 + 6 * off)
        if score > best_score:
            best, best_score = m, score
    if best is None:
        raise RuntimeError("no object-sized blob fully in view")
    best = cv2.morphologyEx(best.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(best)
    return lab_img == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def color_box(color):
    """Plain background: pixels that differ from the border colour; biggest component near the centre."""
    h, w = color.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(color, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, min(h, w) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    med = np.median(border, 0)
    spread = np.percentile(np.linalg.norm(border - med, axis=1), 90)
    diff = np.linalg.norm(lab - med, axis=2)
    fg = (diff > max(18.0, 1.5 * spread)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab_img, stats, cents = cv2.connectedComponentsWithStats(fg)
    best, best_score = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.002 * h * w or (bw > 0.95 * w and bh > 0.95 * h):
            continue
        off = np.hypot((cents[i][0] - w / 2) / w, (cents[i][1] - h / 2) / h)
        score = area / (1 + 4 * off)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        raise RuntimeError("no foreground found by colour")
    return lab_img == best


def sam_mask(color, box):
    from capture_to_3d import make_mask

    m = make_mask(color, box).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)  # keep the biggest piece
    if n > 1:
        m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # fill holes
    cv2.drawContours(m, contours, -1, 1, cv2.FILLED)
    return m.astype(bool)


def auto_mask(color, depth, cam, method, max_depth):
    h, w = color.shape[:2]
    rough, used = None, None
    for m in (["depth", "color"] if method == "auto" else [method]):
        try:
            rough = depth_box(color, depth, cam, max_depth) if m == "depth" else color_box(color)
            used = m
            break
        except (RuntimeError, SystemExit, ValueError) as e:
            print(f"  {m} box failed: {e}")
    if rough is None:
        sys.exit("Could not find the object automatically. Is it in view, on a plain background?")
    box = box_of(rough, 8, w, h)
    print(f"Auto box ({used}): {box}")
    mask = sam_mask(color, box)
    iou = (mask & rough).sum() / max((mask | rough).sum(), 1)
    print(f"SAM 2 mask: {mask.sum()} px, IoU with the rough {used} mask {iou:.2f}")
    if iou < 0.3:
        print("WARNING: SAM 2 and the rough mask disagree a lot, using the rough mask.")
        mask = rough
    return mask, box, used


def preview(color, mask, box):
    vis = color.copy()
    vis[mask] = (0.5 * vis[mask] + [0, 127, 0]).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(vis, tuple(box[:2]), tuple(box[2:]), (0, 200, 255), 2)
    return vis


# ================================================================ 3. SAM 3D with the real depth
def reconstruct(color, depth, mask, cam, seed, max_depth):
    import torch
    from capture_to_3d import SAM3D_REPO

    sys.path.append(os.path.join(SAM3D_REPO, "notebook"))
    from inference import Inference
    from sam3d_lowvram import enable_low_vram

    # SAM 3D's point maps use the PyTorch3D camera convention (x left, y up, z forward),
    # RealSense / OpenCV is (x right, y down, z forward).
    pts, valid = backproject(depth, cam, max_depth)
    pm = pts.astype(np.float32) * np.array([-1, -1, 1], np.float32)
    pm[~valid] = np.nan
    pointmap = torch.from_numpy(pm)

    print("Loading SAM 3D (~20 s)...")
    cwd = os.getcwd()
    os.chdir(SAM3D_REPO)
    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
    os.chdir(cwd)
    enable_low_vram(inference, verbose=False)
    print("Building the 3D model (~40 s)...")
    t0 = time.time()
    out = inference(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), mask, seed=seed, pointmap=pointmap)
    print(f"Done in {time.time() - t0:.0f} s")
    return out


def to_mesh(glb):
    import trimesh

    if isinstance(glb, trimesh.Scene):
        return glb.dump(concatenate=True)
    return glb


def simplify(mesh, faces):
    """SAM 3D meshes have ~400k+ vertices; FoundationPose renders hundreds of hypotheses at once and
    runs out of GPU memory on that. Quadric decimation (open3d) keeps the shape and vertex colours."""
    import open3d as o3d
    import trimesh

    if len(mesh.faces) <= faces:
        return mesh
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh.vertices),
                                  o3d.utility.Vector3iVector(mesh.faces))
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices):
        m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vc)[:, :3] / 255.0)
    m = m.simplify_quadric_decimation(target_number_of_triangles=faces)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    colors = (np.asarray(m.vertex_colors) * 255).astype(np.uint8) if m.has_vertex_colors() else None
    return trimesh.Trimesh(np.asarray(m.vertices), np.asarray(m.triangles), vertex_colors=colors, process=False)


def metric_mesh(out, depth, mask, cam, max_depth, mode="auto"):
    """Scale the normalised SAM 3D mesh to metres. Two estimates:
      sam3d: SAM 3D's own layout scale (it got the real RealSense point map), ~10-20 % off in tests
      depth: longest visible side of the object's depth points = the mesh's longest side; accurate when
             that side is seen side-on (e.g. a can lying down, a box from above), too small otherwise.
    auto uses depth when the longest side is side-on (or SAM 3D is way off), otherwise sam3d."""
    import trimesh

    mesh = to_mesh(out["glb"]).copy()
    mesh.apply_translation(-mesh.bounding_box.centroid)

    s_sam = float(np.asarray(out["scale"].float().cpu()).reshape(-1)[:3].mean())

    # Depth check: the visible surface's size. Its largest extent can't exceed the object's diagonal and
    # is usually close to the object's largest dimension.
    pts, valid = backproject(depth, cam, max_depth)
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & valid
    P = pts[m]
    z = np.median(P[:, 2])
    P = P[np.abs(P[:, 2] - z) < 0.25]  # drop flying pixels
    c = P - P.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt.T
    vis_ext = np.percentile(proj, 98, 0) - np.percentile(proj, 2, 0)
    vis_max = float(vis_ext.max())

    s_depth = vis_max / float(mesh.extents.max())  # assumes the object's longest side is the visible one
    # Is the visible longest side seen side-on (lying across the view)? Then its depth length is reliable.
    view_dir = P.mean(0) / np.linalg.norm(P.mean(0))
    side_on = abs(float(vt[0] @ view_dir)) < 0.5  # longest visible axis > 60 deg from the viewing ray
    ratio = s_sam / s_depth
    print(f"Scale: SAM 3D {s_sam:.4f} (longest side {100 * s_sam * mesh.extents.max():.1f} cm), "
          f"depth {s_depth:.4f} (longest visible side {100 * vis_max:.1f} cm, side-on={side_on})")
    if mode == "sam3d" or mode == "depth":
        how = mode
    elif not 0.6 <= ratio <= 1.8:
        how = "depth"
        print("WARNING: SAM 3D's size disagrees a lot with the depth, using the depth.")
    else:
        how = "depth" if side_on else "sam3d"
    scale = s_depth if how == "depth" else s_sam
    candidates = {"sam3d": s_sam, "depth": s_depth}
    mesh.apply_scale(scale)
    return mesh, {"scale": scale, "scale_source": how, "scale_candidates": candidates,
                  "extents_m": mesh.extents.tolist(), "visible_extent_m": vis_ext.tolist()}


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="object name -> ~/my_scripts/objects/<name>/")
    ap.add_argument("--countdown", type=float, default=3.0, help="seconds of live view before the photo")
    ap.add_argument("--no-window", action="store_true", help="no live-view window (headless)")
    ap.add_argument("--input", help="re-use a saved capture folder (color.png, depth.png, camera.json)")
    ap.add_argument("--box-method", choices=["auto", "depth", "color"], default="auto")
    ap.add_argument("--max-depth", type=float, default=1.5, help="ignore depth beyond this (m)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30, help="use 15 if 30 fails on USB 2")
    ap.add_argument("--scale", choices=["auto", "depth", "sam3d"], default="auto",
                    help="metric size from the depth points or from SAM 3D (auto picks, see info.json)")
    ap.add_argument("--faces", type=int, default=20000, help="simplify the saved mesh to this many triangles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--view", action="store_true", help="open the 3D viewer at the end")
    args = ap.parse_args()

    out_dir = os.path.join(OBJECTS, args.name)
    os.makedirs(out_dir, exist_ok=True)

    if args.input:
        color, depth, cam = load_capture(os.path.expanduser(args.input))
    else:
        color, depth, cam = capture(args.width, args.height, args.fps, args.countdown, not args.no_window)
    if os.path.abspath(os.path.expanduser(args.input or "")) != os.path.abspath(out_dir):
        save_capture(out_dir, color, depth, cam)

    mask, box, used = auto_mask(color, depth, cam, args.box_method, args.max_depth)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(out_dir, "preview.png"), preview(color, mask, box))

    import torch
    torch.cuda.empty_cache()
    out = reconstruct(color, depth, mask, cam, args.seed, args.max_depth)
    to_mesh(out["glb"]).export(os.path.join(out_dir, "object_sam3d_raw.glb"))
    out["gs"].save_ply(os.path.join(out_dir, "object.ply"))

    mesh, info = metric_mesh(out, depth, mask, cam, args.max_depth, args.scale)
    n_raw = len(mesh.vertices)
    mesh = simplify(mesh, args.faces)
    print(f"Mesh simplified: {n_raw} -> {len(mesh.vertices)} vertices")
    mesh.export(os.path.join(out_dir, "mesh.obj"))
    mesh.export(os.path.join(out_dir, "mesh.glb"))
    info.update({"name": args.name, "box": box, "box_method": used, "camera": cam})
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    e = np.array(info["extents_m"]) * 100
    print(f"\nSaved to {out_dir}\n  mesh.obj / mesh.glb : {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} cm "
          f"(size from {info['scale_source']}; --scale depth|sam3d to override)")
    print(f"  Track it:  conda activate foundationpose && python {HERE}/track_object.py {args.name}")

    if args.view:
        from view_mesh import show
        show(os.path.join(out_dir, "mesh.glb"))


if __name__ == "__main__":
    main()
