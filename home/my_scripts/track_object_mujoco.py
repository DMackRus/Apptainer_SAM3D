import argparse
import glob
import os
import sys
import time
import numpy as np

import mujoco
import mujoco.viewer
import cv2

os.environ.pop("SESSION_MANAGER", None)

# Camera pose in the MuJoCo world (4x4, camera -> world). Camera frame is OpenCV: x right, y down, z forward.
# Placeholder: camera 0.6 m above the world origin looking straight down (image "up" = world +y).
# Replace with your real camera extrinsics (measured, or from a hand-eye / checkerboard calibration).
T_WORLD_CAM = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.6],
    [0.0,  0.0,  0.0, 1.0],
])

HERE = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.expanduser("~/FoundationPose")
OBJECTS = os.path.join(HERE, "objects")
WIN = "track_object (FoundationPose)"
sys.path.insert(0, HERE)

def find_mesh(name):
    if os.path.isfile(os.path.expanduser(name)):
        return os.path.expanduser(name)
    d = os.path.join(OBJECTS, name)
    for f in ("mesh.obj", "mesh.glb"):
        if os.path.isfile(os.path.join(d, f)):
            return os.path.join(d, f)
    have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(OBJECTS, "*")))
    sys.exit(f"No mesh for '{name}' in {d}. Scanned objects: {have or 'none'} "
             f"(make one with: python {HERE}/scan_object.py {name})")

def detect(bgr, raw_depth, cam, method, max_depth):
    """Object mask on this frame, no clicking (scan_object's auto box + SAM 2)."""
    import torch
    from scan_object import auto_mask

    mask, box, used = auto_mask(bgr, raw_depth, cam, method, max_depth)
    torch.cuda.empty_cache()
    return mask


def lost(pose, K, w, h, prev, max_jump):
    """Cheap sanity checks: object in front of the camera, centre on screen, no teleporting."""
    t = pose[:3, 3]
    if not np.all(np.isfinite(pose)) or t[2] < 0.05 or t[2] > 3.0:
        return True
    u, v = (K @ t)[:2] / t[2]
    if not (-0.1 * w < u < 1.1 * w and -0.1 * h < v < 1.1 * h):
        return True
    return prev is not None and np.linalg.norm(t - prev[:3, 3]) > max_jump

class Camera:
    def __init__(self, width, height, fps):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        prof = self.pipeline.start(cfg)
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.spatial = rs.spatial_filter()
        self.temporal, self.holes = rs.temporal_filter(), rs.hole_filling_filter()
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.cam = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                    "width": intr.width, "height": intr.height, "depth_scale": self.depth_scale}
        self.K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        for _ in range(30):  # auto-exposure
            self.pipeline.wait_for_frames()

    def read(self):
        """BGR uint8, depth uint16 (raw), depth float metres."""
        f = self.align.process(self.pipeline.wait_for_frames(5000))
        d = self.spatial.process(f.get_depth_frame())
        bgr = np.asanyarray(f.get_color_frame().get_data()).copy()
        raw = np.asanyarray(d.get_data()).copy()
        return bgr, raw, raw.astype(np.float32) * self.depth_scale

    def read_filtered(self, n=10):
        """Like read(), but depth averaged over n frames + holes filled (cleaner for finding the object;
        the object should hold still for ~0.3 s)."""
        for _ in range(n):
            f = self.align.process(self.pipeline.wait_for_frames(5000))
            d = self.holes.process(self.temporal.process(self.spatial.process(f.get_depth_frame())))
        bgr = np.asanyarray(f.get_color_frame().get_data()).copy()
        raw = np.asanyarray(d.get_data()).copy()
        return bgr, raw, raw.astype(np.float32) * self.depth_scale

    def stop(self):
        self.pipeline.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("object", help="object name in ~/my_scripts/objects/, or a mesh file (metres)")
    ap.add_argument("--register-iters", type=int, default=5)
    ap.add_argument("--track-iters", type=int, default=2)
    ap.add_argument("--box-method", choices=["auto", "depth", "color"], default="auto",
                    help="how the object is found on the first frame")
    ap.add_argument("--max-depth", type=float, default=1.5)
    ap.add_argument("--max-jump", type=float, default=0.15, help="re-register if it jumps > this (m) in one frame")
    ap.add_argument("--axis-length", type=float, default=0, help="axis length (m), default = 0.6 x object size")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--record", nargs="?", const="auto", metavar="FILE.mp4",
                    help="save the annotated video (no file name = objects/<name>/track_<time>.mp4)")
    # ap.add_argument("--save-poses", help="write every pose (frame time + 4x4 flattened) to this .txt")


    # Load MuJoCo xml
    model = mujoco.MjModel.from_xml_path("mujoco_model/test.xml")
    data = mujoco.MjData(model)
    # qpos / qvel slots of the can's free joint (qpos: x y z qw qx qy qz, qvel: 6)
    jnt = model.body("Tomato_can").jntadr[0]
    qadr, vadr = model.jnt_qposadr[jnt], model.jnt_dofadr[jnt]
    viewer = mujoco.viewer.launch_passive(model, data)

    args = ap.parse_args()

    mesh_path = find_mesh(args.object)

    # Open the OpenCV window BEFORE importing torch/FoundationPose: torchvision loads PyAV, whose bundled
    # libraries make OpenCV's Qt windows hang forever (even Ctrl+C) if the window is created afterwards.
    # splash = np.zeros((args.height, args.width, 3), np.uint8)
    # cv2.putText(splash, "Loading FoundationPose...", (20, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.8, (0, 255, 255), 2)
    # cv2.imshow(WIN, splash)
    # cv2.waitKey(1)

    # FoundationPose imports (its repo isn't a package; it also needs its own cwd-independent paths)
    sys.path.insert(0, FP_DIR)
    import nvdiffrast.torch as dr
    import trimesh
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
    import logging

    set_logging_format()
    logging.getLogger().setLevel(logging.WARNING)  # FoundationPose is very chatty at INFO
    set_seed(0)

    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh.extents.max() > 1.0:
        print(f"WARNING: mesh is {mesh.extents.max():.2f} units big - FoundationPose needs metres.")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    axis_len = args.axis_length or 0.6 * float(extents.max())
    print(f"Mesh {mesh_path}: {len(mesh.vertices)} verts, size {np.round(extents * 100, 1)} cm")

    debug_dir = os.path.join("/tmp", f"foundationpose_{os.getuid()}")
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         glctx=dr.RasterizeCudaContext(), debug=0, debug_dir=debug_dir)

    camera = Camera(args.width, args.height, args.fps)
    K, w, h = camera.K, args.width, args.height
    # writer = None
    # if args.record:
    #     if args.record == "auto":
    #         folder = os.path.dirname(mesh_path)
    #         args.record = os.path.join(folder, time.strftime("track_%Y%m%d_%H%M%S.mp4"))
    #     args.record = os.path.abspath(os.path.expanduser(args.record))
    #     if not os.path.splitext(args.record)[1]:
    #         args.record += ".mp4"  # OpenCV needs the extension to pick a format, else it silently writes nothing
    #     writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    #     if not writer.isOpened():
    #         print(f"WARNING: can't write video to {args.record}, recording disabled.")
    #         writer = None
    #     else:
    #         print(f"Recording to {args.record}")

    pose, need_register, show_axes, show_box = None, True, True, True
    t_last_print, n, fps = 0.0, 0, 0.0
    t_start = time.time()
    try:
        while True:
            bgr, raw, depth = camera.read_filtered() if need_register else camera.read()
            rgb = bgr[..., ::-1].copy()
            depth[(depth < 0.05) | (depth > args.max_depth)] = 0
            t0 = time.time()
            status = "tracking"
            if need_register:
                try:
                    mask = detect(bgr, raw, camera.cam, args.box_method, args.max_depth)
                    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.register_iters)
                    need_register = False
                    status = "registered"
                    print(f"Registered in {time.time() - t0:.1f} s")
                except SystemExit as e:  # object not found this frame: keep trying
                    print(f"Detection failed ({e}), retrying...")
                    pose = None
            else:
                new = est.track_one(rgb=rgb, depth=depth, K=K, iteration=args.track_iters)
                if lost(new, K, w, h, pose, args.max_jump):
                    print("Tracking lost -> re-registering on the next frame")
                    need_register, status = True, "lost"
                pose = new
            dt = time.time() - t0
            fps = 0.9 * fps + 0.1 / max(dt, 1e-3) if fps else 1 / max(dt, 1e-3)

            # Set the pose of the object to MuJoCo frame
            # pose = object -> camera (OpenCV camera: x right, y down, z forward), metres, in the mesh.obj frame.
            # MuJoCo keeps the body frame at the OBJ's origin, so body pose in world = T_WORLD_CAM @ pose.
            if pose is not None and status != "lost":
                T = T_WORLD_CAM @ pose
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, np.ascontiguousarray(T[:3, :3]).reshape(-1))  # -> w, x, y, z
                data.qpos[qadr:qadr + 3] = T[:3, 3]
                data.qpos[qadr + 3:qadr + 7] = quat
                data.qvel[vadr:vadr + 6] = 0  # kinematic: driven by the camera, not by physics

            mujoco.mj_forward(model, data)  # (mujoco.forward doesn't exist)
            viewer.sync()  # push the new pose to the passive viewer

            # vis = bgr.copy()
            # if pose is not None and status != "lost":
            #     center = pose @ np.linalg.inv(to_origin)  # pose of the oriented bounding box centre
            #     if show_box:
            #         vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center, bbox=bbox, line_color=(0, 255, 0))
            #     if show_axes:
            #         vis = draw_xyz_axis(vis, ob_in_cam=center, scale=axis_len, K=K, thickness=3,
            #                             transparency=0, is_input_rgb=False)
            #     t = pose[:3, 3]
            #     txt = f"{status}  {fps:4.1f} Hz  xyz = {t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f} m"
            #     if pose_file:
            #         pose_file.write(f"{time.time():.3f} " + " ".join(f"{x:.6f}" for x in pose.reshape(-1)) + "\n")
            #     if time.time() - t_last_print > 1.0:
            #         rpy = np.array(cv2.RQDecomp3x3(pose[:3, :3])[0])  # already in degrees
            #         print(f"[{fps:4.1f} Hz] t = {np.round(t, 3)} m, rot xyz = {np.round(rpy, 1)} deg")
            #         t_last_print = time.time()
            # else:
            #     txt = "looking for the object..."
            # cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            # cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            # cv2.putText(vis, "r = re-detect  a = axes  b = box  q = quit", (10, h - 12),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # if writer:
            #     writer.write(vis)
            # n += 1
            # if not args.no_window:
            #     cv2.imshow(WIN, vis)
            #     key = cv2.waitKey(1) & 0xFF
            #     if key in (ord("q"), 27):
            #         break
            #     if key == ord("r"):
            #         need_register = True
            #     if key == ord("a"):
            #         show_axes = not show_axes
            #     if key == ord("b"):
            #         show_box = not show_box
            # if args.frames and n >= args.frames:
            #     break
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        # if writer:
        #     writer.release()
        # if pose_file:
        #     pose_file.close()
        cv2.destroyAllWindows()
    print(f"{n} frames in {time.time() - t_start:.0f} s")

if __name__ == "__main__":
    main()