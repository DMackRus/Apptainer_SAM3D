#!/usr/bin/env python
"""
RealSense RGB-D photo -> segment the obvious object -> cylinder or box? -> simple primitive mesh.

    python ~/my_scripts/image_to_primitive.py                     # live view, SPACE takes the photo
    python ~/my_scripts/image_to_primitive.py --name mug
    python ~/my_scripts/image_to_primitive.py --input ~/sam3d_outputs/mug   # re-run on a saved capture

Assumes: one object standing upright on a flat surface (table/floor), plain background,
camera looking at it from the side or from above (a downward angle of ~20-60 deg works best,
because then both the top and the sides are visible).

How it works:
  1. Capture colour + depth (aligned to colour, filtered over a few frames).
  2. Find the support plane (table) with RANSAC on the depth point cloud; the object is the biggest
     blob of points sticking up out of it. SAM 2 then sharpens the mask using that blob's box.
  3. Classify, using the surface normals of the visible object points:
       - sides: a box's side normals all point in 1-2 directions 90 deg apart, a cylinder's are
         spread around the axis.  -> "side squareness" in [0, 1]
       - top:   a box's top face fills its bounding rectangle (~1.0), a disk fills pi/4 (~0.79).
                                                                 -> "top squareness" in [0, 1]
  4. Fit the primitive: height from the table, cylinder radius from a circle fit (the depth tells us
     how far back the centre is), box size/orientation from the face normals. Parts the camera can't
     see (e.g. the depth of a box seen face-on) are assumed (square footprint, see --box-depth).

Saved in ~/sam3d_outputs/<name>/:
    color.png, depth.png (uint16), camera.json   the capture (re-usable with --input)
    mask.png, primitive_overlay.png              segmentation + primitive drawn on the photo
    object_points.ply                            the object's 3D points (camera frame)
    primitive.obj                                primitive mesh in its own frame (centred, z = up)
    primitive_in_camera.obj                      same mesh placed in the camera frame
    primitive.json                               type, dimensions (m), 4x4 pose camera->object, scores
"""
import argparse
import json
import os
import sys
import time

os.environ.pop("SESSION_MANAGER", None)  # silences a harmless Qt warning from OpenCV windows

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.expanduser("~/sam3d_outputs")
SAM2_CHECKPOINT = os.path.expanduser("~/checkpoints/sam2.1_hiera_small.pt")


# ================================================================ 1. capture
def capture(width, height, fps, preview=True):
    """Returns colour (BGR), depth (uint16, aligned to colour), camera dict."""
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
        d = frames.get_depth_frame()
        d = holes.process(temporal.process(spatial.process(d)))  # smoother depth -> better normals
        return frames.get_color_frame(), d

    try:
        for _ in range(30):  # auto-exposure settles, temporal filter fills up
            grab()
        if preview:
            print("Live view: press SPACE to take the photo, q to quit.")
            while True:
                c, d = grab()
                img = np.asanyarray(c.get_data())
                dv = cv2.applyColorMap(cv2.convertScaleAbs(np.asanyarray(d.get_data()), alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("RealSense - SPACE = take photo, q = quit", np.hstack([img, dv]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if key in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    sys.exit("Quit.")
            cv2.destroyAllWindows()
            for _ in range(10):  # hold still: let the temporal filter average the final view
                c, d = grab()
        else:
            c, d = grab()
        intr = d.profile.as_video_stream_profile().get_intrinsics()  # aligned -> colour intrinsics
        cam = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
               "width": intr.width, "height": intr.height, "depth_scale": depth_scale}
        return np.asanyarray(c.get_data()).copy(), np.asanyarray(d.get_data()).copy(), cam
    finally:
        pipeline.stop()


def save_capture(out_dir, color, depth, cam):
    cv2.imwrite(os.path.join(out_dir, "color.png"), color)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth.astype(np.uint16))
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(cam, f, indent=2)


def load_capture(d):
    color = cv2.imread(os.path.join(d, "color.png"))
    depth = cv2.imread(os.path.join(d, "depth.png"), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(d, "camera.json")) as f:
        cam = json.load(f)
    if color is None or depth is None:
        sys.exit(f"{d} needs color.png, depth.png and camera.json")
    return color, depth, cam


# ================================================================ 2. geometry helpers
def backproject(depth, cam, max_depth):
    """Pixel-aligned (H, W, 3) point map in metres (camera frame: x right, y down, z forward) + valid mask."""
    h, w = depth.shape
    z = depth.astype(np.float64) * cam["depth_scale"]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.dstack([(u - cam["cx"]) * z / cam["fx"], (v - cam["cy"]) * z / cam["fy"], z])
    return pts, (z > 0.05) & (z < max_depth)


def project(pts, cam):
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[:, 0] * cam["fx"] / pts[:, 2] + cam["cx"],
                     pts[:, 1] * cam["fy"] / pts[:, 2] + cam["cy"]], axis=1)


class Support:
    """Support plane: height(x) = up . x + d  (metres above the table)."""

    def __init__(self, up, d, found):
        self.up, self.d, self.found = np.asarray(up, float), float(d), found
        # In-plane axes: camera x projected onto the plane, and up x e1.
        e1 = np.array([1.0, 0, 0]) - self.up[0] * self.up
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.up, self.e1)

    def height(self, p):
        return p @ self.up + self.d

    def to2d(self, p):
        return np.stack([p @ self.e1, p @ self.e2], axis=-1)

    def to3d(self, xy, h):
        return xy[0] * self.e1 + xy[1] * self.e2 + (h - self.d) * self.up


def depth_sigma(z):
    """Rough RealSense D4xx depth noise (m) at range z (m): ~1 mm at 0.5 m, ~4 mm at 1 m, ~16 mm at 2 m."""
    return 0.004 * np.asarray(z) ** 2


def find_planes(pcd, max_planes=3):
    """Returns (support plane or None, [background planes]) as (normal, d) with the camera on the + side."""
    planes, rest, n0 = [], pcd, len(pcd.points)
    for _ in range(max_planes):
        if len(rest.points) < 0.1 * n0:
            break
        (a, b, c, d), inl = rest.segment_plane(distance_threshold=0.008, ransac_n=3, num_iterations=1000)
        if len(inl) < 0.12 * n0:
            break
        n = np.array([a, b, c])
        if d < 0:  # camera (origin) on the positive side
            n, d = -n, -d
        planes.append((n / np.linalg.norm(n), d / np.linalg.norm(n), len(inl)))
        rest = rest.select_by_index(inl, invert=True)
    # Support = the plane whose normal points most "up" in the image (camera -y). With an upright
    # camera tilted down by e, the table normal has -y = cos^2(e) >= 0 while a wall behind the object
    # has -y = -sin(e)cos(e) <= 0, so this also works for steep top-down views.
    ups = [p for p in planes if -p[0][1] > -0.05]
    support = max(ups, key=lambda p: -p[0][1]) if ups else None
    background = [p for p in planes if p is not support]
    return support, background


# ================================================================ 3. segmentation
def segment(color, pts, valid, cam, use_sam=True, verbose=True):
    import open3d as o3d
    from scipy.spatial import cKDTree

    flat = pts[valid]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(flat)).voxel_down_sample(0.005)
    support_p, background = find_planes(pcd)

    P = np.asarray(pcd.points)
    tol = 0.008 + 3 * depth_sigma(P[:, 2])  # stereo depth noise grows with z^2
    keep = np.ones(len(P), bool)
    for n, d, _ in background:
        keep &= (P @ n + d) > tol  # in front of the wall, not on/behind it
    if support_p is not None:
        support = Support(support_p[0], support_p[1], True)
        hgt = support.height(P)
        keep &= (hgt > 0.006 + 2 * depth_sigma(P[:, 2])) & (hgt < 1.0)
    else:
        print("WARNING: no table plane found, assuming the image's up direction is 'up'.")
        support = Support([0, -1.0, 0], 0.0, False)
    cand = P[keep]
    if len(cand) < 50:
        sys.exit("No object found above the table. Is the object in view and >1 cm tall?")

    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
                        .cluster_dbscan(eps=0.015, min_points=10))
    best, best_score = None, -1
    for lab in set(labels) - {-1}:
        c = cand[labels == lab]
        uv = project(c.mean(0, keepdims=True), cam)[0]
        off = np.hypot((uv[0] - cam["width"] / 2) / cam["width"], (uv[1] - cam["height"] / 2) / cam["height"])
        score = len(c) / (1 + 3 * off)  # big and central wins
        if score > best_score:
            best, best_score = c, score
    if best is None:
        sys.exit("Couldn't cluster an object out of the depth.")

    # Pixel mask: valid pixels within 1 cm of the chosen blob.
    dist, _ = cKDTree(best).query(pts.reshape(-1, 3), distance_upper_bound=0.01)
    mask = (dist < np.inf).reshape(valid.shape) & valid
    if support.found:
        mask &= support.height(pts) > 0.004
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n > 1:
        mask = lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = mask.astype(bool)

    if use_sam and os.path.exists(SAM2_CHECKPOINT):
        try:
            sys.path.insert(0, HERE)
            from capture_to_3d import make_mask
            ys, xs = np.nonzero(mask)
            pad = 6
            box = [max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                   min(xs.max() + pad, cam["width"] - 1), min(ys.max() + pad, cam["height"] - 1)]
            sam = make_mask(color, box)
            iou = (sam & mask).sum() / max((sam | mask).sum(), 1)
            if verbose:
                print(f"SAM 2 mask vs depth mask IoU {iou:.2f}")
            if iou > 0.5:  # trust SAM's sharper edges, but drop table/background pixels inside it
                near, _ = cKDTree(best).query(pts.reshape(-1, 3), distance_upper_bound=0.02)
                mask = sam & valid & (near < np.inf).reshape(valid.shape)
                if support.found:
                    mask &= support.height(pts) > 0.004
            else:
                print("SAM 2 disagrees with the depth segmentation, keeping the depth mask.")
        except Exception as e:  # noqa: BLE001
            print(f"SAM 2 refinement skipped ({type(e).__name__}: {e})")

    if not support.found:  # put the 'table' at the object's lowest point
        support.d = -np.percentile(pts[mask] @ support.up, 1)
    return mask, support


# ================================================================ 4. classify + fit
def object_cloud(pts, mask):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[mask]))
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.012, max_nn=40))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


def circle_fit(xy):
    """Algebraic (Kasa) least-squares circle: returns centre, radius, rms residual."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r = np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-12))
    res = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r
    return np.array([cx, cy]), r, float(np.sqrt((res ** 2).mean()))


def analyse(pcd, support, force=None, box_depth=None):
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    hgt = support.height(P)
    xy = support.to2d(P)
    n_up = N @ support.up
    top = n_up > 0.85
    side = np.abs(n_up) < 0.5
    H = float(np.percentile(hgt, 99))

    # Horizontal viewing direction (camera -> object) in plane coords, and the visible width across it.
    centroid = np.median(xy, 0)
    view = centroid / (np.linalg.norm(centroid) + 1e-9)  # camera is at (0, 0) in plane coords
    perp = np.array([-view[1], view[0]])
    across = xy @ perp
    vis_width = float(np.percentile(across, 99) - np.percentile(across, 1))

    # --- side evidence: box face normals agree modulo 90 deg -> |mean(exp(4i*theta))| ~ 1
    side_sq, theta0 = None, None
    hn = support.to2d(N[side])
    w = np.linalg.norm(hn, axis=1)
    if side.sum() >= 100:
        theta = np.arctan2(hn[:, 1], hn[:, 0])
        z = (w * np.exp(4j * theta)).sum() / w.sum()
        r4 = float(np.abs(z))
        theta0 = float(np.angle(z) / 4)
        side_sq = float(np.clip((r4 - 0.25) / (0.65 - 0.25), 0, 1))
    # --- top evidence: convex hull area / min-area-rectangle area (box ~1.0, disk ~0.785)
    top_sq, fill = None, None
    if top.sum() >= 100:
        t2 = xy[top].astype(np.float32)
        hull_area = cv2.contourArea(cv2.convexHull(t2))
        (_, _), (rw, rh), ang = cv2.minAreaRect(t2)
        fill = hull_area / max(rw * rh, 1e-12)
        top_sq = float(np.clip((fill - 0.80) / (0.94 - 0.80), 0, 1))
        if theta0 is None:
            theta0 = float(np.deg2rad(ang))

    ws = [(s, n) for s, n in ((side_sq, side.sum()), (top_sq, 1.5 * top.sum())) if s is not None]
    if not ws:
        sys.exit("Too few clean object points to decide the shape (move closer / improve lighting).")
    squareness = sum(s * n for s, n in ws) / sum(n for _, n in ws)
    kind = force or ("box" if squareness > 0.5 else "cylinder")
    notes = []

    if kind == "cylinder":
        fits = []
        if top.sum() >= 100:  # the top disk is the most reliable: smallest circle around it
            (cx, cy), r = cv2.minEnclosingCircle(xy[top].astype(np.float32))
            fits.append(("top disk", np.array([cx, cy]), float(r)))
        if side.sum() >= 100:  # circle through the visible arc of the side wall
            c, r, rms = circle_fit(xy[side])
            if rms < 0.15 * r and 0.3 * vis_width < r < 1.5 * vis_width:
                fits.append(("side arc", c, r))
        if fits:
            how, center, radius = fits[0]
        else:  # can't see enough curvature: radius = half the visible width, centre pushed back
            radius = vis_width / 2
            front = np.percentile(xy @ view, 2)
            center = (front + radius) * view + np.median(across) * perp
            how = "assumed from visible width"
        notes.append(f"radius from {how}")
        dims = {"radius": float(radius), "height": H}
        x_axis = support.e1
    else:
        if theta0 is None:
            theta0 = 0.0
        a = np.array([np.cos(theta0), np.sin(theta0)])
        b = np.array([-a[1], a[0]])
        pa, pb = xy @ a, xy @ b
        lo = np.array([np.percentile(pa, 1), np.percentile(pb, 1)])
        hi = np.array([np.percentile(pa, 99), np.percentile(pb, 99)])
        ext = hi - lo
        # A box seen face-on with no top visible shows no depth: extend it away from the camera.
        thin = int(np.argmin(ext))
        if box_depth is not None or (ext[thin] < 0.35 * ext[1 - thin] and top.sum() < 0.15 * len(P)):
            new = box_depth if box_depth is not None else ext[1 - thin]  # default: square footprint
            away = np.sign((a, b)[thin] @ view) or 1.0
            if away > 0:
                hi[thin] = lo[thin] + new
            else:
                lo[thin] = hi[thin] - new
            notes.append(f"box depth not visible, assumed {new * 100:.1f} cm")
            ext = hi - lo
        center = (lo + hi)[0] / 2 * a + (lo + hi)[1] / 2 * b
        dims = {"size_x": float(ext[0]), "size_y": float(ext[1]), "height": H}
        x_axis = a[0] * support.e1 + a[1] * support.e2

    z_axis = support.up
    y_axis = np.cross(z_axis, x_axis)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = support.to3d(center, H / 2)  # object frame origin = centre of the primitive
    scores = {"squareness": round(squareness, 3),
              "side_squareness": None if side_sq is None else round(side_sq, 3),
              "top_squareness": None if top_sq is None else round(top_sq, 3),
              "top_fill_ratio": None if fill is None else round(fill, 3),
              "n_points": int(len(P)), "n_side": int(side.sum()), "n_top": int(top.sum())}
    return kind, dims, T, scores, notes


def make_mesh(kind, dims):
    import trimesh

    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=dims["radius"], height=dims["height"], sections=64)
    return trimesh.creation.box(extents=[dims["size_x"], dims["size_y"], dims["height"]])


def wire_lines(kind, dims):
    """Primitive outline as 3D line segments in the object frame."""
    h = dims["height"] / 2
    segs = []
    if kind == "cylinder":
        t = np.linspace(0, 2 * np.pi, 49)
        ring = np.column_stack([dims["radius"] * np.cos(t), dims["radius"] * np.sin(t)])
        for z in (-h, h):
            segs += [(np.r_[ring[i], z], np.r_[ring[i + 1], z]) for i in range(48)]
        segs += [(np.r_[ring[i], -h], np.r_[ring[i], h]) for i in range(0, 48, 6)]
    else:
        sx, sy = dims["size_x"] / 2, dims["size_y"] / 2
        c = np.array([[x, y, z] for z in (-h, h) for x, y in ((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy))])
        for i in range(4):
            segs += [(c[i], c[(i + 1) % 4]), (c[4 + i], c[4 + (i + 1) % 4]), (c[i], c[4 + i])]
    return segs


def draw_overlay(color, mask, kind, dims, T, cam):
    vis = color.copy()
    vis[mask] = (0.6 * vis[mask] + [0, 100, 0]).astype(np.uint8)
    for a, b in wire_lines(kind, dims):
        pa, pb = (T[:3, :3] @ np.stack([a, b]).T).T + T[:3, 3]
        if pa[2] <= 0.01 or pb[2] <= 0.01:
            continue
        (u0, v0), (u1, v1) = project(np.stack([pa, pb]), cam)
        cv2.line(vis, (int(u0), int(v0)), (int(u1), int(v1)), (0, 0, 255), 2, cv2.LINE_AA)
    label = f"{kind}: " + ", ".join(f"{k}={v * 100:.1f}cm" for k, v in dims.items())
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def show_3d(pcd, color, pts, mask, kind, dims, T):
    import open3d as o3d

    scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[~mask & (pts[..., 2] > 0)]))
    scene.colors = o3d.utility.Vector3dVector(color[~mask & (pts[..., 2] > 0)][:, ::-1] / 255.0)
    obj = o3d.geometry.PointCloud(pcd)
    obj.paint_uniform_color([0.1, 0.8, 0.1])
    segs = wire_lines(kind, dims)
    P = np.array([p for s in segs for p in s])
    P = (T[:3, :3] @ P.T).T + T[:3, 3]
    lines = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P),
                                 o3d.utility.Vector2iVector(np.arange(len(P)).reshape(-1, 2)))
    lines.paint_uniform_color([1, 0, 0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * dims["height"]).transform(T)
    print("3D view: scene points, object points (green), primitive (red). Drag = rotate, q = close.")
    o3d.visualization.draw_geometries([scene, obj, lines, frame], window_name="primitive fit",
                                      front=[0, 0, -1], up=[0, -1, 0], lookat=T[:3, 3].tolist(), zoom=0.5)


# ================================================================ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="output folder name (default: capture time)")
    parser.add_argument("--input", help="re-use a saved capture folder (color.png, depth.png, camera.json)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="use 15 if 30 fails on USB 2")
    parser.add_argument("--max-depth", type=float, default=2.0, help="ignore depth beyond this (m)")
    parser.add_argument("--force", choices=["cylinder", "box"], help="skip classification")
    parser.add_argument("--box-depth", type=float, help="box depth (m) when the camera can't see it")
    parser.add_argument("--no-sam", action="store_true", help="depth-only segmentation (no SAM 2)")
    parser.add_argument("--no-preview", action="store_true", help="take the photo straight away")
    parser.add_argument("--no-view", action="store_true", help="don't open any windows at the end")
    args = parser.parse_args()

    if args.input:
        in_dir = os.path.expanduser(args.input)
        color, depth, cam = load_capture(in_dir)
        out_dir = os.path.join(OUTPUTS, args.name) if args.name else in_dir
    else:
        out_dir = os.path.join(OUTPUTS, args.name or time.strftime("prim_%Y%m%d_%H%M%S"))
        color, depth, cam = capture(args.width, args.height, args.fps, preview=not args.no_preview)
    os.makedirs(out_dir, exist_ok=True)
    if not args.input or out_dir != os.path.expanduser(args.input):
        save_capture(out_dir, color, depth, cam)

    import open3d as o3d
    import trimesh

    pts, valid = backproject(depth, cam, args.max_depth)
    mask, support = segment(color, pts, valid, cam, use_sam=not args.no_sam)
    pcd = object_cloud(pts, mask)
    kind, dims, T, scores, notes = analyse(pcd, support, args.force, args.box_depth)

    print(f"\n==> {kind.upper()}  " + "  ".join(f"{k}={v * 100:.1f} cm" for k, v in dims.items()))
    print(f"    squareness {scores['squareness']:.2f} (0 = cylinder, 1 = box; "
          f"sides {scores['side_squareness']}, top {scores['top_squareness']})")
    for n in notes:
        print(f"    note: {n}")
    if not support.found:
        print("    note: no table plane, height measured from the object's lowest visible point")

    mesh = make_mesh(kind, dims)
    mesh.export(os.path.join(out_dir, "primitive.obj"))
    mesh.copy().apply_transform(T).export(os.path.join(out_dir, "primitive_in_camera.obj"))
    o3d.io.write_point_cloud(os.path.join(out_dir, "object_points.ply"), pcd)
    cv2.imwrite(os.path.join(out_dir, "mask.png"), mask.astype(np.uint8) * 255)
    overlay = draw_overlay(color, mask, kind, dims, T, cam)
    cv2.imwrite(os.path.join(out_dir, "primitive_overlay.png"), overlay)
    with open(os.path.join(out_dir, "primitive.json"), "w") as f:
        json.dump({"type": kind, "dimensions_m": dims, "T_camera_object": T.round(6).tolist(),
                   "table_plane": {"normal": support.up.round(6).tolist(), "d": round(support.d, 6),
                                   "found": support.found},
                   "scores": scores, "notes": notes}, f, indent=2)
    print(f"Saved to {out_dir}: primitive.obj / primitive_in_camera.obj / primitive.json / primitive_overlay.png")

    if not args.no_view:
        cv2.imshow("Primitive fit - any key = 3D view", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        show_3d(pcd, color, pts, mask, kind, dims, T)


if __name__ == "__main__":
    main()
