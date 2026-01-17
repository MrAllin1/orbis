#!/usr/bin/env python3
"""
Make a random validation split by MOVING N .mp4 videos from one folder to another.

Default behavior:
- source: data/covla_videos_balanced
- dest:   data/covla_videos_balanced_val
- N:      412

It saves a manifest of moved files in:
  data/covla_videos_balanced_val/val_manifest.txt

Usage:
  python make_covla_val_split.py

Optional args:
  --src /path/to/src
  --dst /path/to/dst
  --n 412
  --seed 1337
  --copy   (copy instead of move)
"""

import argparse
import os
import random
import shutil
from pathlib import Path


def list_mp4_files(src: Path) -> list[Path]:
    files = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/covla_videos_balanced", help="Source directory containing .mp4 files")
    ap.add_argument("--dst", default="data/covla_videos_balanced_val", help="Destination directory for validation .mp4 files")
    ap.add_argument("--n", type=int, default=412, help="Number of videos to move/copy into validation")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"[ERROR] Source folder does not exist: {src}")

    videos = list_mp4_files(src)
    if len(videos) == 0:
        raise SystemExit(f"[ERROR] No .mp4 files found in: {src}")

    if args.n > len(videos):
        raise SystemExit(f"[ERROR] Requested n={args.n} but only found {len(videos)} videos in {src}")

    random.seed(args.seed)
    chosen = random.sample(videos, args.n)

    manifest_path = dst / "val_manifest.txt"
    op = shutil.copy2 if args.copy else shutil.move

    moved = 0
    for p in chosen:
        target = dst / p.name
        if target.exists():
            raise SystemExit(f"[ERROR] Destination file already exists (refusing to overwrite): {target}")
        op(str(p), str(target))
        moved += 1

    # Write manifest (one filename per line)
    with manifest_path.open("w", encoding="utf-8") as f:
        for p in sorted([c.name for c in chosen]):
            f.write(p + "\n")

    print("[DONE]")
    print(f"  Source:      {src}")
    print(f"  Dest:        {dst}")
    print(f"  Operation:   {'COPY' if args.copy else 'MOVE'}")
    print(f"  Seed:        {args.seed}")
    print(f"  Selected:    {moved} / {args.n}")
    print(f"  Manifest:    {manifest_path}")


if __name__ == "__main__":
    main()
