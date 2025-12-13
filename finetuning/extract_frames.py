#!/usr/bin/env python
import argparse
import os
import random
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image


def pick_video(path: Path) -> Path:
    """
    If `path` is a file, return it.
    If `path` is a directory, randomly pick one .mp4 file inside.
    """
    if path.is_file():
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"File is not an .mp4: {path}")
        return path

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    mp4_files = sorted([p for p in path.iterdir() if p.suffix.lower() == ".mp4"])
    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found in directory: {path}")

    return random.choice(mp4_files)


def extract_frames(video_path: Path, out_dir: Path, num_frames: int = 5):
    """
    Extract `num_frames` frames from `video_path` and save as PNGs into `out_dir`.
    Frames are sampled roughly evenly across the whole video.
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Video has zero frames: {video_path}")

    # If video has fewer than num_frames, just take all frames
    n = min(num_frames, total)

    # Evenly spaced indices between 0 and total-1
    indices = np.linspace(0, total - 1, num=n, dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving {n} frames from {video_path} to {out_dir}")

    for i, idx in enumerate(indices):
        frame = vr[idx].asnumpy()          # H x W x 3 (uint8)
        img = Image.fromarray(frame)
        save_path = out_dir / f"frame_{i:02d}.png"
        img.save(save_path)
        print(f"  [OK] Saved frame {idx} -> {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract 5 frames from a random mp4 in a given path."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to an .mp4 file OR a directory containing .mp4 files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for frames. "
             "Default: <video_stem>_frames next to the video.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=5,
        help="Number of frames to extract (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (for picking the video if a directory is given).",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    path = Path(args.path).resolve()
    video_path = pick_video(path)

    if args.out_dir is None:
        out_dir = video_path.parent / f"{video_path.stem}_frames"
    else:
        out_dir = Path(args.out_dir).resolve()

    extract_frames(video_path, out_dir, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
