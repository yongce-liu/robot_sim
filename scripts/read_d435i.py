#!/usr/bin/env python3
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

from robot_sim.utils.saver import write_records


@dataclass
class Args:
    """Read RGB/depth frames from Intel RealSense D435i."""

    width: int = 640  # Color/depth width.
    height: int = 480  # Color/depth height.
    fps: int = 30  # Stream FPS.
    max_frames: int = 0  # Stop after N frames (0 = run until quit).
    save_dir: Path = Path("outputs/debug/realsense/d435i")  # Save frames to this directory.
    no_display: bool = False  # Disable live display window.
    save_image: bool = True  # Save images to disk.


def main() -> None:
    args = tyro.cli(Args)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.save_image:
        (args.save_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (args.save_dir / "depth").mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    frame_idx = 0
    start_time = time.time()
    rgb_frames = []
    depth_frames = []
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            depth_m = depth.astype(np.float32) * depth_scale
            depth_vis = np.clip(depth_m / 2.0, 0.0, 1.0)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

            if args.save_dir is not None and args.save_image:
                cv2.imwrite(str(args.save_dir / f"rgb/{frame_idx:06d}.png"), color)
                cv2.imwrite(str(args.save_dir / f"depth/{frame_idx:06d}.png"), depth)
                rgb_frames.append(color)
                depth_frames.append(depth_vis)

            if not args.no_display:
                stacked = np.hstack((color, depth_vis))
                cv2.imshow("D435i RGB + Depth", stacked)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        pipeline.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        write_records(path=args.save_dir / "rgb_output.mp4", records=rgb_frames, fps=args.fps)
        write_records(path=args.save_dir / "depth_output.mp4", records=depth_frames, fps=args.fps)

    elapsed = time.time() - start_time
    fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"Captured {frame_idx} frames in {elapsed:.2f}s ({fps:.1f} FPS).")


if __name__ == "__main__":
    main()
