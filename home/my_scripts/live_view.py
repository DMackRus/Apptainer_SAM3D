#!/usr/bin/env python
"""
Live RealSense view. Runs until you press Ctrl+C in the terminal (or q / Esc in the window).

    python ~/my_scripts/live_view.py              # colour only
    python ~/my_scripts/live_view.py --depth      # colour + colourised depth side by side
    python ~/my_scripts/live_view.py --fps 15     # USB 2: use 15 if 30 fails
"""
import argparse
import os
import time

os.environ.pop("SESSION_MANAGER", None)  # silences a harmless Qt warning from OpenCV windows

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30, help="use 15 if 30 fails on USB 2")
    parser.add_argument("--depth", action="store_true", help="also show the depth stream")
    args = parser.parse_args()

    if not rs.context().query_devices():
        raise SystemExit("No RealSense device found.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipeline.start(config)
    align = rs.align(rs.stream.color) if args.depth else None

    window = "RealSense live (Ctrl+C or q to quit)"
    print("Streaming. Press Ctrl+C here (or q in the window) to stop.")
    n, t0 = 0, time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames(5000)
            if align:
                frames = align.process(frames)
            img = np.asanyarray(frames.get_color_frame().get_data())
            if args.depth:
                depth = np.asanyarray(frames.get_depth_frame().get_data())
                img = np.hstack([img, cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)])
            cv2.imshow(window, img)
            n += 1
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        dt = time.time() - t0
        print(f"\nStopped. {n} frames in {dt:.0f} s ({n / max(dt, 1e-6):.1f} fps).")


if __name__ == "__main__":
    main()
