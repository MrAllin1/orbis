#!/usr/bin/env python3
"""
make_rollout_montage.py

Creates a single comparison animation from multiple rollout variants, plus (optionally)
a ground-truth MP4 shown as a smaller video tile in the grid.

Expected structure (NEW, after multi-prompt rollout loop):
  <root_dir>/<variant>/run_*/combined_context_plus_gen.gif
  <root_dir>/<variant>/run_*/caption.txt

Fallback for older runs:
  <root_dir>/<variant>/run_*/combined_context_plus_gen.gif
  <root_dir>/<variant>/run_*/<something>.out  (caption line near end)

Outputs (scrubbable):
  - comparison.mp4
  - comparison.html

Usage:
  python make_rollout_montage.py /path/to/second_video --ncols 3 --fps 5

With ground-truth MP4:
  python make_rollout_montage.py /path/to/second_video \
    --real_video /work/.../goes_straight_video_2/0a4ee25856dc0338.mp4 \
    --real_scale 0.65 \
    --ncols 3 --fps 5 --max_frames 120
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAPTION_PATTERNS = [
    re.compile(r"^\s*Caption used\s*:\s*(.*)\s*$"),
    re.compile(r"^\s*Caption configured\s*:\s*(.*)\s*$"),
]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        return s[1:-1]
    return s


def extract_caption_from_caption_txt(run_dir: Path) -> Optional[str]:
    """
    NEW: primary caption source for your updated rollout script.
    Expects: <run_dir>/caption.txt
    """
    p = run_dir / "caption.txt"
    if not p.exists():
        return None
    try:
        txt = p.read_text(errors="ignore").strip()
    except Exception:
        return None
    if not txt:
        return None
    if txt.upper() in {"<NONE>", "NONE"}:
        return None
    return txt


def extract_caption_from_out(out_path: Path) -> Optional[str]:
    try:
        lines = out_path.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    for line in reversed(lines):
        for pat in CAPTION_PATTERNS:
            m = pat.match(line)
            if m:
                raw = _strip_quotes(m.group(1).strip())
                if raw.upper() in {"<NONE>", "NONE"}:
                    return None
                return raw
    return None


def pick_newest_run_dir(variant_dir: Path) -> Optional[Path]:
    """
    Choose newest run_* directory by modification time (more robust than name sorting).
    """
    run_dirs = [p for p in variant_dir.glob("run_*") if p.is_dir()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime)
    return run_dirs[-1]


def find_run_assets(root_dir: Path) -> List[Tuple[str, Path, Optional[str]]]:
    """
    Returns list of (variant_name, gif_path, caption).
    Chooses the newest run_* directory per variant.

    NEW layout supported:
      <root>/<variant>/run_*/combined_context_plus_gen.gif
      <root>/<variant>/run_*/caption.txt
    """
    items: List[Tuple[str, Path, Optional[str]]] = []

    # Only consider immediate subdirectories as variants
    for variant_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        run_dir = pick_newest_run_dir(variant_dir)
        if run_dir is None:
            continue

        gif_path = run_dir / "combined_context_plus_gen.gif"
        if not gif_path.exists():
            continue

        # NEW: caption.txt first
        caption = extract_caption_from_caption_txt(run_dir)

        # Fallback: parse latest *.out
        if caption is None:
            out_files = sorted(run_dir.glob("*.out"), key=lambda p: p.stat().st_mtime)
            caption = extract_caption_from_out(out_files[-1]) if out_files else None

        items.append((variant_dir.name, gif_path, caption))

    return items


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def load_gif_frames(gif_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    with imageio.get_reader(str(gif_path)) as reader:
        for i, frm in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(_to_rgb_uint8(np.asarray(frm)))
    if not frames:
        raise RuntimeError(f"No frames loaded from: {gif_path}")
    return frames


def load_mp4_frames(mp4_path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    with imageio.get_reader(str(mp4_path)) as reader:
        for i, frm in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(_to_rgb_uint8(np.asarray(frm)))
    if not frames:
        raise RuntimeError(f"No frames loaded from: {mp4_path}")
    return frames


def pick_common_frame_count(all_frames: List[List[np.ndarray]], max_frames: Optional[int]) -> int:
    n = min(len(x) for x in all_frames)
    if max_frames is not None:
        n = min(n, max_frames)
    return max(1, n)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        # draw.textlength exists in modern Pillow; fallback safely if needed
        try:
            ok = draw.textlength(trial, font=font) <= max_width
        except Exception:
            ok = len(trial) <= max(10, max_width // max(6, getattr(font, "size", 12)))
        if ok:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def make_caption_block(width: int, height: int, title: str, caption: Optional[str], font: ImageFont.ImageFont) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    pad_x, pad_y = 10, 6
    y = pad_y

    draw.text((pad_x, y), title, fill=(0, 0, 0), font=font)
    y += getattr(font, "size", 16) + 4

    cap = caption if caption else "<NO CAPTION>"
    cap = f"Caption: {cap}"
    for line in wrap_text(draw, cap, font, max_width=width - 2 * pad_x)[:5]:
        draw.text((pad_x, y), line, fill=(0, 0, 0), font=font)
        y += getattr(font, "size", 16) + 2

    return img


def resize_and_pad(frame: np.ndarray, target_w: int, target_h: int, scale: float) -> Image.Image:
    scale = float(scale)
    scale = max(0.05, min(scale, 1.0))

    img = Image.fromarray(frame, mode="RGB")
    w, h = img.size

    fit_scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * fit_scale * scale))
    new_h = max(1, int(h * fit_scale * scale))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas.paste(img, (x0, y0))
    return canvas


def compose_grid_frame(
    tiles: List[np.ndarray],
    cap_imgs: List[Image.Image],
    scales: List[float],
    ncols: int,
    tile_w: int,
    tile_h: int,
    cap_h: int,
    pad: int,
) -> np.ndarray:
    n = len(tiles)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    out_w = ncols * tile_w + (ncols + 1) * pad
    out_h = nrows * (tile_h + cap_h) + (nrows + 1) * pad

    canvas = Image.new("RGB", (out_w, out_h), (245, 245, 245))

    for idx in range(n):
        r, c = divmod(idx, ncols)
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + cap_h + pad)

        tile_img = resize_and_pad(tiles[idx], tile_w, tile_h, scales[idx])
        canvas.paste(tile_img, (x0, y0))
        canvas.paste(cap_imgs[idx], (x0, y0 + tile_h))

    return np.asarray(canvas)


def write_html_viewer(html_path: Path, mp4_path: Path, entries: List[Tuple[str, Optional[str]]]):
    rel_mp4 = mp4_path.name
    lines = []
    for variant, cap in entries:
        cap_txt = cap if cap else "<NO CAPTION>"
        lines.append(f"<li><b>{variant}</b><br/><span>{cap_txt}</span></li>")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Rollout Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    video {{ width: 100%; max-width: 1200px; border: 1px solid #ccc; }}
    ul {{ max-width: 1200px; }}
    li {{ margin: 10px 0; }}
    span {{ color: #222; }}
  </style>
</head>
<body>
  <h2>Rollout Comparison</h2>
  <p>Use the controls to pause/rewind/scrub.</p>
  <video controls preload="metadata">
    <source src="{rel_mp4}" type="video/mp4"/>
    Your browser does not support MP4 playback.
  </video>
  <h3>Captions</h3>
  <ul>
    {''.join(lines)}
  </ul>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root_dir", type=str, help="Root directory containing variant subfolders.")
    ap.add_argument("--real_video", type=str, default=None, help="Optional ground-truth MP4 to include as a small tile.")
    ap.add_argument("--real_scale", type=float, default=0.65, help="Scale for ground-truth tile inside its box.")
    ap.add_argument("--ncols", type=int, default=3)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--caption_height", type=int, default=90)
    ap.add_argument("--pad", type=int, default=14)
    ap.add_argument("--tile_w", type=int, default=None)
    ap.add_argument("--tile_h", type=int, default=None)
    ap.add_argument("--write_gif", action="store_true", help="Also write comparison.gif (optional fallback).")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    runs = find_run_assets(root_dir)
    if not runs:
        raise RuntimeError(
            f"No runs found under {root_dir}.\n"
            "Expected: <root>/<variant>/run_*/combined_context_plus_gen.gif"
        )

    # Load rollout GIF frames
    variant_names = [x[0] for x in runs]
    gif_paths = [x[1] for x in runs]
    captions = [x[2] for x in runs]
    rollout_frames = [load_gif_frames(p, max_frames=args.max_frames) for p in gif_paths]

    # Optional: load real video frames and insert as first tile
    all_frames: List[List[np.ndarray]] = rollout_frames
    all_names: List[str] = variant_names
    all_caps: List[Optional[str]] = captions
    scales: List[float] = [1.0] * len(rollout_frames)

    if args.real_video is not None:
        real_path = Path(args.real_video).expanduser().resolve()
        if not real_path.exists():
            raise FileNotFoundError(f"real_video not found: {real_path}")
        real_frames = load_mp4_frames(real_path, max_frames=args.max_frames)

        all_frames = [real_frames] + rollout_frames
        all_names = ["GROUND_TRUTH"] + variant_names
        all_caps = ["real video"] + captions
        scales = [args.real_scale] + [1.0] * len(rollout_frames)

    common_n = pick_common_frame_count(all_frames, max_frames=args.max_frames)

    # Tile size from first source
    h0, w0 = all_frames[0][0].shape[:2]
    tile_w = args.tile_w if args.tile_w else w0
    tile_h = args.tile_h if args.tile_h else h0

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    cap_imgs = [
        make_caption_block(tile_w, args.caption_height, name, cap, font)
        for name, cap in zip(all_names, all_caps)
    ]

    out_frames: List[np.ndarray] = []
    for t in range(common_n):
        tiles_t = [frames[t] for frames in all_frames]
        out_frames.append(
            compose_grid_frame(
                tiles=tiles_t,
                cap_imgs=cap_imgs,
                scales=scales,
                ncols=args.ncols,
                tile_w=tile_w,
                tile_h=tile_h,
                cap_h=args.caption_height,
                pad=args.pad,
            )
        )

    mp4_path = root_dir / "comparison.mp4"
    html_path = root_dir / "comparison.html"
    gif_path = root_dir / "comparison.gif"

    # MP4 (scrubbable)
    try:
        imageio.mimsave(str(mp4_path), out_frames, fps=args.fps)
        print(f"[OK] Wrote MP4 (scrubbable): {mp4_path}")
    except Exception as e:
        print(f"[WARN] Failed to write MP4 (ffmpeg/plugin missing?): {e}")
        print("[WARN] You can still use --write_gif to get a GIF output.")

    # HTML viewer
    if mp4_path.exists():
        write_html_viewer(html_path, mp4_path, list(zip(all_names, all_caps)))
        print(f"[OK] Wrote HTML viewer: {html_path}")

    # Optional GIF fallback
    if args.write_gif:
        imageio.mimsave(str(gif_path), out_frames, fps=args.fps, loop=0)
        print(f"[OK] Wrote GIF: {gif_path}")

    print("\n[FOUND TILES]")
    for name, cap in zip(all_names, all_caps):
        print(f" - {name}: {cap if cap else '<NO CAPTION>'}")


if __name__ == "__main__":
    main()
