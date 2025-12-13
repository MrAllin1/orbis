#!/usr/bin/env python
import os
import time
from pathlib import Path

import torch
import torchvision.utils as vutils
import imageio
import numpy as np
from PIL import Image
from torchvision import transforms

from finetuning.adaln_config import (
    load_fm_config,
    ORBIT_CKPT,
    SAVE_PATH,        # finetuned AdaLN checkpoint
)
from finetuning.adaln_models import (
    create_text_encoder,
    create_world_model,
)

# ------------------------------------------------------------------------------------
# CONFIG – EDIT THESE
# ------------------------------------------------------------------------------------
DESIRED_SECONDS = 20          # target total clip length in seconds
BASE_OUT_DIR = "finetuning/inference_samples_try3-no-promptt"

FRAMES_DIR = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/rollout_frames/5frames"  # <--- EDIT
TEXT_PROMPT = None  # <--- EDIT


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor (C,H,W) in [-1,1] or [0,1] to uint8 HWC.
    """
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    # If in [-1, 1], map to [0, 1]
    if t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)

    # CHW -> HWC
    t = t.detach().float().cpu()
    t = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return t.numpy()


def load_frames_from_folder(frames_dir: str, size_hw, device: torch.device) -> torch.Tensor:
    """
    Load all images from frames_dir, sort by name, resize to size_hw=(H,W),
    convert to tensor in [-1,1], and stack as (1, F, 3, H, W).
    """
    p = Path(frames_dir)
    if not p.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    # collect image files
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])

    if len(files) == 0:
        raise RuntimeError(f"No image files found in {frames_dir}")

    print(f"[INFO] Found {len(files)} frames in: {frames_dir}")

    H, W = size_hw
    tfm = transforms.Compose(
        [
            transforms.Resize((H, W)),
            transforms.ToTensor(),          # [0,1]
        ]
    )

    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        t = tfm(img)          # (3, H, W) in [0,1]
        t = t * 2.0 - 1.0     # → [-1,1]
        frames.append(t)

    frames = torch.stack(frames, dim=0)   # (F, 3, H, W)
    frames = frames.unsqueeze(0).to(device)  # (1, F, 3, H, W)
    return frames


def main():
    # ---------------- DEVICE ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---------------- LOAD CONFIG & DATA PARAMS ----------------
    # load_fm_config returns: fm_cfg, model_cfg, train_params, val_params
    fm_cfg, model_cfg, train_params, val_params = load_fm_config()

    # Get resolution and frame_rate from training config
    size = train_params["size"]          # e.g. [192, 336]
    H, W = int(size[0]), int(size[1])
    frame_rate = float(train_params.get("target_frame_rate", 5.0))

    print(f"[INFO] Using frame size: {H}x{W}")
    print(f"[INFO] Using target_frame_rate: {frame_rate} fps")

    # ---------------- LOAD CONTEXT FRAMES FROM FOLDER ----------------
    images = load_frames_from_folder(FRAMES_DIR, (H, W), device)  # (1, F, 3, H, W)
    _, F, _, _, _ = images.shape
    video_id = Path(FRAMES_DIR).name

    print(f"\n[INFO] Frames folder name (video_id): {video_id}")
    print(f"[INFO] Number of context frames provided: {F}")
    print(f"[INFO] Hardcoded caption: '{TEXT_PROMPT}'")

    # ---------------- BUILD MODELS ----------------
    # 1) CLIP-based text encoder
    text_encoder = create_text_encoder(device)

    # 2) FM world model (Stage2 ModelIF)
    max_frames = train_params["num_frames"]  # e.g. 6; passed but not critical
    world_model = create_world_model(model_cfg, max_frames, device, ORBIT_CKPT)
    world_model.eval()

    # ---------------- LOAD FINETUNED ADALN CHECKPOINT ----------------
    print(f"\n[INFO] Loading finetuned AdaLN checkpoint from: {SAVE_PATH}")
    ckpt = torch.load(SAVE_PATH, map_location=device)

    # Load AdaLN + vit + ae etc.
    state_dict = ckpt["state_dict"]
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded finetuned world model weights.")
    print(f"       Missing keys   : {len(missing)}")
    print(f"       Unexpected keys: {len(unexpected)}")

    # Load text projection head
    text_proj_state = ckpt["text_proj"]
    text_encoder.proj.load_state_dict(text_proj_state, strict=True)
    print("[INFO] Loaded finetuned text projection weights.\n")

    # ---------------- PREPARE OUTPUT FOLDER ----------------
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(BASE_OUT_DIR) / f"run_{run_id}_{video_id}"
    context_dir = out_dir / "context_frames"
    gen_dir = out_dir / "gen_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving results to: {out_dir}")

    # ---------------- ENCODE + ROLL OUT ----------------
    with torch.no_grad():
        # Encode caption
        text_emb = text_encoder([TEXT_PROMPT])  # (1, 768)

        # Encode frames to latents
        latents = world_model.encode_frames(images)  # (1, F, E, H', W')
        B, F_lat, E, H_lat, W_lat = latents.shape
        print(f"[INFO] Encoded latents shape: {latents.shape}")

        # Use ALL provided frames as context
        context = latents  # (1, F, E, H', W')
        context_frames = context.shape[1]
        print(f"[INFO] Using {context_frames} context frames.")

        # Compute how many frames for ~DESIRED_SECONDS total
        total_frames_needed = int(DESIRED_SECONDS * frame_rate)
        num_gen_frames = max(total_frames_needed - context_frames, 1)

        print(f"[INFO] Target clip length: {DESIRED_SECONDS}s")
        print(f"[INFO] => total_frames_needed = {total_frames_needed}")
        print(f"[INFO] => num_gen_frames to generate = {num_gen_frames}")

        # Roll out num_gen_frames conditioned on text
        _, decoded = world_model.roll_out(
            x_0=context,
            num_gen_frames=num_gen_frames,
            latent_input=True,
            eta=0.0,
            NFE=30,
            sample_with_ema=True,
            num_samples=images.size(0),
            text_emb=text_emb,
        )
        # decoded: (1, num_gen_frames, 3, H, W) in [-1,1]
        print(f"[INFO] Generated frames shape: {decoded.shape}")

    # ---------------- SAVE CONTEXT + GENERATED FRAMES (PNGs) ----------------
    # Save original context frames
    for t in range(images.shape[1]):
        frame = images[0, t]  # (3, H, W), in [-1,1]
        frame_vis = (frame + 1.0) / 2.0
        vutils.save_image(
            frame_vis,
            context_dir / f"context_{t:02d}.png"
        )

    # Save generated frames
    for t in range(decoded.shape[1]):
        frame = decoded[0, t]  # (3, H, W), in [-1,1]
        frame_vis = (frame + 1.0) / 2.0
        vutils.save_image(
            frame_vis,
            gen_dir / f"gen_{t:02d}.png"
        )

    # ---------------- BUILD GIFs ----------------
    # 1) Combined GIF: context frames + generated frames
    combined_frames = []

    for t in range(images.shape[1]):
        combined_frames.append(images[0, t])
    for t in range(decoded.shape[1]):
        combined_frames.append(decoded[0, t])

    combined_gif_frames = [to_uint8_image(frm) for frm in combined_frames]
    combined_gif_path = out_dir / "combined_context_plus_gen.gif"
    imageio.mimsave(combined_gif_path, combined_gif_frames, fps=int(frame_rate), loop=0)
    print(f"[GIF] Saved combined GIF: {combined_gif_path}")

    # 2) Generated-only GIF
    gen_only_frames = [decoded[0, t] for t in range(decoded.shape[1])]
    gen_only_gif_frames = [to_uint8_image(frm) for frm in gen_only_frames]
    gen_only_gif_path = out_dir / "generated_only.gif"
    imageio.mimsave(gen_only_gif_path, gen_only_gif_frames, fps=int(frame_rate), loop=0)
    print(f"[GIF] Saved generated-only GIF: {gen_only_gif_path}")

    print("\n[DONE]")
    print(f"Context PNGs in      : {context_dir}")
    print(f"Generated PNGs in    : {gen_dir}")
    print(f"Combined GIF         : {combined_gif_path}")
    print(f"Generated-only GIF   : {gen_only_gif_path}")
    print(f"Frames folder (video_id) : {video_id}")
    print(f"Caption used         : '{TEXT_PROMPT}'")


if __name__ == "__main__":
    main()
