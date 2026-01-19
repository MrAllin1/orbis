#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu

# You must have fvd_metric.py (or fvd_metric/) next to this script or on PYTHONPATH
from fvd_metric import compute_fv


def list_mp4s(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    folder_str = str(folder)
    if "real_by_class" in folder_str:
        mp4s = sorted([p for p in folder.rglob("*.mp4")])
    else:
        mp4s = sorted([p for p in folder.rglob("generated_only.mp4")])
    print(f"[INFO] Found {len(mp4s)} mp4 files under: {folder}")
    if not mp4s:
        raise RuntimeError(f"No mp4 files found under: {folder}")
    return mp4s


def sample_indices(total: int, num_frames: int) -> np.ndarray:
    # Evenly sample num_frames across the clip
    if total <= 0:
        raise RuntimeError("Video has 0 frames")
    if total >= num_frames:
        return np.linspace(0, total - 1, num=num_frames, dtype=np.int64)
    # If too short, pad by repeating last frame
    idx = np.arange(total, dtype=np.int64)
    pad = np.full((num_frames - total,), total - 1, dtype=np.int64)
    return np.concatenate([idx, pad], axis=0)


def load_video_tensor(
    video_path: Path,
    num_frames: int,
    size_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Returns: (3, T, H, W) float32 in [0,1]
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    idx = sample_indices(total, num_frames)

    frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3), uint8
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)

    H, W = size_hw
    frames_t = torch.nn.functional.interpolate(
        frames_t, size=(H, W), mode="bilinear", align_corners=False
    )

    # -> (3,T,H,W)
    frames_t = frames_t.permute(1, 0, 2, 3).contiguous()
    return frames_t


def load_set(
    folder: Path,
    max_videos: int,
    num_frames: int,
    size_hw: Tuple[int, int],
    seed: int,
) -> torch.Tensor:
    """
    Returns: (N, 3, T, H, W)
    """
    mp4s = list_mp4s(folder)

    rng = np.random.default_rng(seed)
    if max_videos > 0 and max_videos < len(mp4s):
        chosen = rng.choice(mp4s, size=max_videos, replace=False)
        mp4s = sorted([Path(p) for p in chosen])

    vids = []
    for p in mp4s:
        vids.append(load_video_tensor(p, num_frames=num_frames, size_hw=size_hw))

    return torch.stack(vids, dim=0)


def resolve_max_items(num_videos: int, max_videos: int, num_samples: int) -> int:
    """
    Backwards-compatible logic:
    - max_videos: subsampling when loading each set (already applied in load_set)
    - num_samples: optional additional cap for FVD computation (maps to max_items in fvd_metric)
    We return a single 'max_items' for compute_fvd.
    """
    candidates = [num_videos]
    if max_videos and max_videos > 0:
        candidates.append(max_videos)
    if num_samples and num_samples > 0:
        candidates.append(num_samples)
    return int(min(candidates))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True, type=str)
    ap.add_argument("--fake_dir", required=True, type=str)

    ap.add_argument(
        "--num_frames",
        default=16,
        type=int,
        help="FVD is commonly computed on fixed-length clips (e.g., 16).",
    )
    ap.add_argument("--height", required=True, type=int)
    ap.add_argument("--width", required=True, type=int)

    ap.add_argument(
        "--max_videos",
        default=0,
        type=int,
        help="0 = use all. Otherwise subsample this many from each set (during loading).",
    )
    ap.add_argument("--seed", default=42, type=int)

    ap.add_argument("--batch_size", default=4, type=int)

    # Backwards-compatible: keep num_samples, but it will map to max_items in compute_fvd
    ap.add_argument(
        "--num_samples",
        default=0,
        type=int,
        help="0 = use all (or max_videos). Otherwise cap FVD computation to this many items (max_items).",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    real_dir = Path(args.real_dir).resolve()
    fake_dir = Path(args.fake_dir).resolve()
    size_hw = (args.height, args.width)

    # Load (subsample here if max_videos>0)
    y_true = load_set(real_dir, args.max_videos, args.num_frames, size_hw, seed=args.seed)
    y_pred = load_set(fake_dir, args.max_videos, args.num_frames, size_hw, seed=args.seed + 999)

    # Match N by truncating to min N
    if y_true.shape[0] != y_pred.shape[0]:
        n = min(y_true.shape[0], y_pred.shape[0])
        y_true = y_true[:n]
        y_pred = y_pred[:n]

    num_videos = int(y_true.shape[0])
    max_items = resolve_max_items(num_videos=num_videos, max_videos=args.max_videos, num_samples=args.num_samples)

    print("[INFO] tensor shape:", tuple(y_true.shape))
    print(f"[INFO] num_videos={num_videos} max_items_for_fvd={max_items} batch_size={args.batch_size}")

    # IMPORTANT: fvd_metric.compute_fvd expects `max_items`, not `num_samples`
    fvd_value = compute_fvd(
        y_true=y_true,
        y_pred=y_pred,
        max_items=max_items if max_items > 0 else None,
        device=device,
        batch_size=args.batch_size,
    )

    print(f"\nFVD(real={real_dir.name}, fake={fake_dir.name}) = {float(fvd_value):.4f}")


if __name__ == "__main__":
    main()
