#!/usr/bin/env python3
import argparse
import random
import re
from pathlib import Path
from collections import defaultdict

LINE_RE = re.compile(
    r"^(?P<file>\S+)\s+(?P<label>left|right|straight)\s+confidence=(?P<conf>[0-9.]+)\s+frames=(?P<frames>\d+)\s+counts=.*$"
)

def parse_labels_file(path: Path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        rows.append({
            "jsonl": m.group("file"),  # e.g. 0019f....jsonl
            "id": Path(m.group("file")).stem,
            "label": m.group("label"),
            "conf": float(m.group("conf")),
            "frames": int(m.group("frames")),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_txt", required=True, type=str, help="Path to covla_video_labels.txt")
    ap.add_argument("--videos_dir", required=True, type=str, help="Path to covla_videos_balanced")
    ap.add_argument("--out_root", required=True, type=str, help="Output root; creates left/right/straight subdirs")
    ap.add_argument("--n_per_class", default=200, type=int, help="How many real videos per class to select")
    ap.add_argument("--min_conf", default=0.85, type=float, help="Minimum confidence threshold")
    ap.add_argument("--min_frames", default=64, type=int, help="Minimum frames per video (for eval clip length safety)")
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--symlink", action="store_true", help="Symlink instead of copying MP4s")
    args = ap.parse_args()

    labels_txt = Path(args.labels_txt).expanduser().resolve()
    videos_dir = Path(args.videos_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = parse_labels_file(labels_txt)
    if not rows:
        raise RuntimeError(f"No parseable rows found in {labels_txt} (regex mismatch?)")

    # filter + bucket
    buckets = defaultdict(list)
    for r in rows:
        if r["conf"] < args.min_conf:
            continue
        if r["frames"] < args.min_frames:
            continue
        mp4 = videos_dir / f"{r['id']}.mp4"
        if not mp4.exists():
            continue
        buckets[r["label"]].append(mp4)

    random.seed(args.seed)
    for cls in ("left", "right", "straight"):
        vids = buckets.get(cls, [])
        if not vids:
            print(f"[WARN] No videos for class={cls} after filters.")
            continue
        random.shuffle(vids)
        chosen = vids[: min(args.n_per_class, len(vids))]

        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for src in chosen:
            dst = out_dir / src.name
            if dst.exists():
                continue
            if args.symlink:
                dst.symlink_to(src)
            else:
                dst.write_bytes(src.read_bytes())

        print(f"[OK] class={cls} selected={len(chosen)} out={out_dir}")

if __name__ == "__main__":
    main()
