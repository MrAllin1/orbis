#!/usr/bin/env python3
"""
Download first 20 CoVLA videos locally as .mp4 files.

Run with:
    conda activate orbis_env
    python download_covla_20_videos.py
"""

import os
from datasets import load_dataset
import cv2

# ===== CONFIG =====
N_VIDEOS = 20
OUT_DIR = "covla_20_videos"
FPS = 20  # CoVLA is 20Hz

# Prefer decord as video backend (same as in the tutorial)
os.environ.setdefault("HF_DATASETS_VIDEO_DECODER", "decord")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] Output directory: {OUT_DIR}")

print("[1/3] Loading CoVLA dataset in streaming mode...")
dataset = load_dataset(
    "turing-motors/CoVLA-Dataset",
    split="train",
    streaming=True,
)

print("[2/3] Iterating over scenes and saving first 20 videos...")

saved = 0
for scene_idx, scene in enumerate(dataset):
    if saved >= N_VIDEOS:
        break

    video = scene["video"]        # decord.VideoReader
    video_id = scene["video_id"]  # e.g. '0000b7dc6478371b'
    output_file = os.path.join(OUT_DIR, f"{video_id}.mp4")

    print(f"\n>>> Scene {scene_idx} (id={video_id})")
    print(f"    Saving to: {output_file}")

    # Get shape from first frame
    first_frame = video[0].asnumpy()
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, FPS, (width, height))

    # Write all frames
    for frame_idx in range(len(video)):
        frame = video[frame_idx].asnumpy()        # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    saved += 1
    print(f"    [OK] Saved video #{saved}: {output_file}")

print(f"\n[3/3] Done. Saved {saved} videos in '{OUT_DIR}'.")
