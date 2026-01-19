#!/usr/bin/env python3
"""
generate_fvd_fake_sets.py

- Samples N videos from a directory (or uses a single mp4 if given).
- Extracts K context frames (0, stride, 2*stride, ...) to PNG cache.
- Runs rollouts for multiple text prompts.
- Writes MP4 outputs (combined + gen_only) under run folders.
- Collects gen_only.mp4 into a clean folder layout for later FVD:

    <collect_root>/<job_tag>/<prompt_key>/
        <job_tag>__<prompt_key>__<video_id>__<run_id>.mp4

Notes:
- This script does NOT need to use the exact same "real" videos as the generated ones for FVD Option C.
  For Option C, you typically compare:
      FVD(real_left_set, fake_left_set), etc.
  The fake_left_set can be generated from any context videos, as long as they are from the same domain and
  have the same frame count / fps / preprocessing as the real sets used in FVD.

Backwards compatibility:
- All existing args remain unchanged.
- New optional flag: --disable_text_conditioning
  If set, the script will ignore all text prompts and run rollouts without text_emb.
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
import imageio.v2 as imageio
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu

from finetuning.adaln_config import load_fm_config, ORBIT_CKPT
from finetuning.adaln_models import create_text_encoder, create_world_model


# ------------------------------------------------------------------------------------
# DEFAULT PROMPTS
# ------------------------------------------------------------------------------------
TEXT_PROMPTS: Dict[str, str] = {
    "turning_left": "Turning left at moderate speed. No traffic lights. Later it slows down.",
    "training_caption": "Moving straight at moderate speed. No traffic lights. Later it slows down.",
    "turning_right": "Turning right at moderate speed. No traffic lights. Later it slows down.",
    "turning_left_right": "Turning left at moderate speed. No traffic lights. Later it turns right.",
    "turning_right_left": "Turning right at moderate speed. No traffic lights. Later it turns left.",
}


# ------------------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------------------
def should_use_text(prompt: str) -> bool:
    return bool(prompt and prompt.strip())


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor (C,H,W) in [-1,1] or [0,1] to uint8 HWC.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    if t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)

    t = t.detach().float().cpu()
    t = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return t.numpy()


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.write_bytes(src.read_bytes())


def write_mp4(frames_uint8_hwc: List[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames_uint8_hwc, fps=fps)
    if not out_path.exists():
        raise RuntimeError(f"MP4 write failed (file not created): {out_path}")


# ------------------------------------------------------------------------------------
# VIDEO PICK + FRAME EXTRACTION (decord)
# ------------------------------------------------------------------------------------
def pick_videos(path: Path, seed: int, num_videos: int, unique: bool = True) -> List[Path]:
    """
    If `path` is a file, returns [path].
    If `path` is a directory, returns num_videos mp4s (unique or with replacement).
    """
    if path.is_file():
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"File is not an .mp4: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    mp4_files = sorted([p for p in path.iterdir() if p.suffix.lower() == ".mp4"])
    if not mp4_files:
        raise RuntimeError(f"No .mp4 files found in directory: {path}")

    rnd = random.Random(seed)

    if unique:
        if num_videos > len(mp4_files):
            raise ValueError(f"Requested num_videos={num_videos} but only {len(mp4_files)} mp4s exist.")
        rnd.shuffle(mp4_files)
        return mp4_files[:num_videos]

    return [rnd.choice(mp4_files) for _ in range(num_videos)]


def extract_context_frames(
    video_path: Path,
    out_dir: Path,
    num_frames: int = 5,
    stride: int = 10,
) -> None:
    """
    Extract frames at indices: 0, stride, 2*stride, ...
    Saves as frame_00.png, frame_01.png, ...
    """
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Video has zero frames: {video_path}")

    max_idx = min(num_frames * stride, total)
    indices = np.arange(0, max_idx, stride, dtype=int)
    if len(indices) == 0:
        indices = np.array([0], dtype=int)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[CTX] Saving {len(indices)} frames from {video_path} -> {out_dir}")

    for i, idx in enumerate(indices):
        frame = vr[idx].asnumpy()
        img = Image.fromarray(frame)
        save_path = out_dir / f"frame_{i:02d}.png"
        img.save(save_path)
        print(f"  [CTX] frame_idx={idx} -> {save_path}")


def load_frames_from_folder(frames_dir: Path, size_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """
    Load frames from frames_dir, sort by name, resize, convert to tensor in [-1,1],
    stack as (1, F, 3, H, W).
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([f for f in frames_dir.iterdir() if f.suffix.lower() in exts])
    if len(files) == 0:
        raise RuntimeError(f"No image files found in {frames_dir}")

    H, W = size_hw
    tfm = transforms.Compose([transforms.Resize((H, W)), transforms.ToTensor()])

    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        t = tfm(img)          # [0,1]
        t = t * 2.0 - 1.0     # [-1,1]
        frames.append(t)

    return torch.stack(frames, dim=0).unsqueeze(0).to(device)  # (1, F, 3, H, W)


# ------------------------------------------------------------------------------------
# INFERENCE PER PROMPT
# ------------------------------------------------------------------------------------
def run_single_prompt(
    *,
    prompt_key: str,
    prompt_text: str,
    desired_seconds: int,
    base_out_dir: Path,
    collect_root: Path,
    job_tag: str,
    frame_rate: float,
    world_model,
    text_encoder,
    images: torch.Tensor,
    video_id: str,
    disable_text_conditioning: bool = False,
    save_png_frames: bool = False,
) -> Path:
    """
    Runs rollout for one prompt, writes MP4s, and collects generated_only.mp4 into:
      collect_root/<job_tag>/<prompt_key>/<job_tag>__<prompt_key>__<video_id>__<run_id>.mp4

    If disable_text_conditioning=True, this function will never compute or pass text_emb.
    """
    use_text = (not disable_text_conditioning) and should_use_text(prompt_text)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = base_out_dir / prompt_key / f"run_{run_id}_{video_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "caption.txt").write_text(prompt_text + "\n", encoding="utf-8")
    (out_dir / "meta.txt").write_text(
        f"prompt_key={prompt_key}\n"
        f"use_text={use_text}\n"
        f"disable_text_conditioning={disable_text_conditioning}\n"
        f"video_id={video_id}\n"
        f"desired_seconds={desired_seconds}\n"
        f"frame_rate={frame_rate}\n"
        f"job_tag={job_tag}\n",
        encoding="utf-8",
    )

    if save_png_frames:
        (out_dir / "context_frames").mkdir(parents=True, exist_ok=True)
        (out_dir / "gen_frames").mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 90)
    print(f"[RUN] prompt_key               = {prompt_key}")
    print(f"[RUN] disable_text_conditioning = {disable_text_conditioning}")
    print(f"[RUN] use_text                 = {use_text}")
    print(f"[RUN] caption                  = '{prompt_text}'")
    print(f"[RUN] out_dir                  = {out_dir}")

    with torch.no_grad():
        text_emb = None
        if use_text and text_encoder is not None:
            text_emb = text_encoder([prompt_text])

        latents = world_model.encode_frames(images)
        context_frames = latents.shape[1]

        total_frames_needed = int(desired_seconds * frame_rate)
        num_gen_frames = max(total_frames_needed - context_frames, 1)

        print(f"[INFO] context_frames={context_frames} total_frames={total_frames_needed} gen_frames={num_gen_frames}")

        rollout_kwargs = dict(
            x_0=latents,
            num_gen_frames=num_gen_frames,
            latent_input=True,
            eta=0.0,
            NFE=30,
            sample_with_ema=True,
            num_samples=images.size(0),
        )
        if use_text and text_emb is not None:
            rollout_kwargs["text_emb"] = text_emb

        _, decoded = world_model.roll_out(**rollout_kwargs)

    if save_png_frames:
        ctx_dir = out_dir / "context_frames"
        gen_dir = out_dir / "gen_frames"

        for t in range(images.shape[1]):
            frame_vis = (images[0, t] + 1.0) / 2.0
            vutils.save_image(frame_vis, ctx_dir / f"context_{t:02d}.png")

        for t in range(decoded.shape[1]):
            frame_vis = (decoded[0, t] + 1.0) / 2.0
            vutils.save_image(frame_vis, gen_dir / f"gen_{t:02d}.png")

    # MP4s
    combined_frames = [images[0, t] for t in range(images.shape[1])] + [
        decoded[0, t] for t in range(decoded.shape[1])
    ]
    combined_mp4_frames = [to_uint8_image(frm) for frm in combined_frames]
    combined_mp4_path = out_dir / "combined_context_plus_gen.mp4"
    write_mp4(combined_mp4_frames, combined_mp4_path, fps=int(frame_rate))

    gen_only_frames = [decoded[0, t] for t in range(decoded.shape[1])]
    gen_only_mp4_frames = [to_uint8_image(frm) for frm in gen_only_frames]
    gen_only_mp4_path = out_dir / "generated_only.mp4"
    write_mp4(gen_only_mp4_frames, gen_only_mp4_path, fps=int(frame_rate))

    print(f"[MP4] combined : {combined_mp4_path}")
    print(f"[MP4] gen_only : {gen_only_mp4_path}")

    # Collect
    dst = collect_root / job_tag / prompt_key / f"{job_tag}__{prompt_key}__{video_id}__{run_id}.mp4"
    safe_copy(gen_only_mp4_path, dst)
    print(f"[COLLECT] {prompt_key} -> {dst}")

    return dst


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sample videos, extract context frames, run multiple prompts, write + collect MP4s for FVD."
    )
    ap.add_argument(
        "--video_source",
        required=True,
        type=str,
        help="Path to an .mp4 OR a directory containing .mp4 files (e.g., covla_videos_balanced).",
    )
    ap.add_argument("--desired_seconds", default=20, type=int)
    ap.add_argument("--num_context_frames", default=5, type=int)
    ap.add_argument("--stride", default=10, type=int, help="Frame stride for context extraction: 0, stride, 2*stride, ...")
    ap.add_argument("--seed", default=42, type=int, help="Seed for selecting videos if a directory is given.")
    ap.add_argument("--num_videos", type=int, default=1, help="How many different context videos to sample.")
    ap.add_argument("--unique_videos", action="store_true", help="Sample without replacement (recommended).")

    ap.add_argument("--base_out_dir", default="finetuning/balanced_captions_dataset_rollouts/second_video/", type=str)
    ap.add_argument(
        "--collect_root",
        default="/work/dlclarge2/alidemaa-text-control-orbis/orbis/fvd_sets/fake_by_prompt",
        type=str,
    )
    ap.add_argument(
        "--job_tag",
        default="modelA",
        type=str,
        help="Tag to isolate outputs per model/job (prevents collisions).",
    )
    ap.add_argument("--save_png_frames", action="store_true", help="Also save PNG context/gen frames under run_dir.")

    ap.add_argument(
        "--use_finetuned_weights",
        action="store_true",
        help="Inject finetuned checkpoint weights (required for your finetuned models).",
    )
    ap.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to finetuned checkpoint (.ckpt). Required if --use_finetuned_weights is set.",
    )

    # NEW (backwards-compatible): disable text conditioning baseline
    ap.add_argument(
        "--disable_text_conditioning",
        action="store_true",
        help="If set, ignores all prompts and runs rollouts without text_emb (baseline).",
    )

    args = ap.parse_args()

    base_out_dir = Path(args.base_out_dir).expanduser().resolve()
    collect_root = Path(args.collect_root).expanduser().resolve()
    collect_root.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load config
    _, model_cfg, train_params, _ = load_fm_config()
    size = train_params["size"]
    H, W = int(size[0]), int(size[1])
    frame_rate = float(train_params.get("target_frame_rate", 5.0))
    print(f"[INFO] Using frame size: {H}x{W}")
    print(f"[INFO] Using target_frame_rate: {frame_rate} fps")

    # Build models (once)
    if args.disable_text_conditioning:
        text_encoder = None
        print("[INFO] Text conditioning DISABLED -> no text encoder will be created.")
    else:
        any_prompt_uses_text = any(should_use_text(p) for p in TEXT_PROMPTS.values())
        text_encoder = create_text_encoder(device) if any_prompt_uses_text else None

    max_frames = train_params["num_frames"]
    world_model = create_world_model(model_cfg, max_frames, device, ORBIT_CKPT)
    world_model.eval()

    # Optional: load finetuned weights (once)
    if args.use_finetuned_weights:
        if not args.save_path:
            raise ValueError("--save_path must be provided when --use_finetuned_weights is set.")

        save_path = Path(args.save_path).expanduser().resolve()
        if not save_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {save_path}")

        print(f"\n[INFO] Loading finetuned AdaLN checkpoint from: {save_path}")
        ckpt = torch.load(str(save_path), map_location=device)
        state_dict = ckpt["state_dict"]

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INJECT][{ts}] Injecting finetuned weights into world_model (strict=False)...")
        missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Injection complete. Missing={len(missing)} Unexpected={len(unexpected)}")

        # Only inject text proj if we actually created a text encoder
        if text_encoder is not None:
            text_proj_state = ckpt["text_proj"]
            ts2 = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INJECT][{ts2}] Injecting finetuned weights into text_encoder.proj (strict=True)...")
            text_encoder.proj.load_state_dict(text_proj_state, strict=True)
            print("[INFO] Text projection injection complete.")
        else:
            if args.disable_text_conditioning:
                print("[INFO] Skipping text projection injection (text conditioning disabled).")
            else:
                print("[INFO] Skipping text projection injection (no text encoder created).")

    # Pick videos
    video_source = Path(args.video_source).expanduser().resolve()
    videos = pick_videos(video_source, seed=args.seed, num_videos=args.num_videos, unique=args.unique_videos)

    print(f"\n[COLLECT] Output root: {collect_root / args.job_tag}")
    print(f"[INFO] Will process {len(videos)} video(s). Each produces {len(TEXT_PROMPTS)} generated MP4s.")
    print(f"[INFO] Total generated MP4s expected: {len(videos) * len(TEXT_PROMPTS)}")
    if args.disable_text_conditioning:
        print("[WARN] Text conditioning disabled: outputs may be identical across prompt keys if rollout is deterministic.")

    # Loop videos
    for i, chosen_video in enumerate(videos, start=1):
        print(f"\n[VIDEO {i}/{len(videos)}] {chosen_video}")
        video_id = chosen_video.stem

        # Cache context frames per video
        ctx_frames_dir = base_out_dir / "_context_frames_cache" / f"{video_id}_seed{args.seed}"
        extract_context_frames(
            chosen_video,
            ctx_frames_dir,
            num_frames=args.num_context_frames,
            stride=args.stride,
        )

        images = load_frames_from_folder(ctx_frames_dir, (H, W), device)
        print(f"[INFO] Loaded context tensor: {tuple(images.shape)} from {ctx_frames_dir}")

        # Run prompts
        for prompt_key, prompt_text in TEXT_PROMPTS.items():
            run_single_prompt(
                prompt_key=prompt_key,
                prompt_text=prompt_text,
                desired_seconds=args.desired_seconds,
                base_out_dir=base_out_dir,
                collect_root=collect_root,
                job_tag=args.job_tag,
                frame_rate=frame_rate,
                world_model=world_model,
                text_encoder=text_encoder,
                images=images,
                video_id=video_id,
                disable_text_conditioning=args.disable_text_conditioning,
                save_png_frames=args.save_png_frames,
            )

    print("\n[ALL DONE]")
    print(f"Collected MP4s under: {collect_root / args.job_tag}")


if __name__ == "__main__":
    main()
