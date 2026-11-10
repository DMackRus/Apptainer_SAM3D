#!/usr/bin/env python
"""Quick RealSense sanity check: lists devices, streams aligned colour + depth for
a few seconds and prints frame statistics. Works in any of the container envs.

    python /scripts/realsense_test.py            # headless check
    python /scripts/realsense_test.py --show     # also show an OpenCV window
"""
import argparse
import time

import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise SystemExit("No RealSense device found.")
    for dev in devices:
        print(f"Found: {dev.get_info(rs.camera_info.name)}  "
              f"serial={dev.get_info(rs.camera_info.serial_number)}  "
              f"fw={dev.get_info(rs.camera_info.firmware_version)}  "
              f"usb={dev.get_info(rs.camera_info.usb_type_descriptor)}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(f"Colour intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    print(f"Depth scale: {depth_scale} m/unit")

    if args.show:
        import cv2

    n = 0
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            frames = align.process(pipeline.wait_for_frames(5000))
            color = np.asanyarray(frames.get_color_frame().get_data())
            depth = np.asanyarray(frames.get_depth_frame().get_data())
            n += 1
            if args.show:
                d = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("realsense (q to quit)", np.hstack([color[..., ::-1], d]))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()

    valid = depth[depth > 0] * depth_scale
    print(f"Received {n} aligned frames in {time.time() - t0:.1f}s; "
          f"colour {color.shape}, depth {depth.shape}; "
          f"valid depth {100 * valid.size / depth.size:.0f}% "
          f"(median {np.median(valid) if valid.size else float('nan'):.2f} m)")
    print("REALSENSE OK" if n > 0 else "REALSENSE FAILED")


if __name__ == "__main__":
    main()
