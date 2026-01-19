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

print("[INFO] Imported necessary modules and the orbis ckpt is at:", ORBIT_CKPT)

# ------------------------------------------------------------------------------------
# CONFIG – EDIT THESE
# ------------------------------------------------------------------------------------
DESIRED_SECONDS = 20
BASE_OUT_DIR = "finetuning/balanced_captions_dataset_rollouts/second_video/"
FRAMES_DIR = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/rollout_frames/goes_straight_video_2"

# NEW: multiple prompts in a map (key -> caption)
TEXT_PROMPTS = {
    "turning_left": "Turning left at moderate speed. No traffic lights. Later it slows down.",
    "training_caption": "Moving straight at moderate speed. No traffic lights. Later it slows down.",
    "turning_right": "Turning right at moderate speed. No traffic lights. Later it slows down.",
    "turning_left_right": "Turning left at moderate speed. No traffic lights. Later it turns right.",
    "turning_right_left": "Turning right at moderate speed. No traffic lights. Later it turns left.",
}

# ------------------------------------------------------------------------------------
# TOGGLES
# ------------------------------------------------------------------------------------
USE_FINETUNED_WEIGHTS = True

# text conditioning is enabled per-prompt if the prompt is non-empty
def should_use_text(prompt: str) -> bool:
    return bool(prompt and prompt.strip())


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    if t.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(t.shape)}")

    if t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)

    t = t.detach().float().cpu()
    t = (t * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
    return t.numpy()


def load_frames_from_folder(frames_dir: str, size_hw, device: torch.device) -> torch.Tensor:
    p = Path(frames_dir)
    if not p.exists():
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])

    if len(files) == 0:
        raise RuntimeError(f"No image files found in {frames_dir}")

    print(f"[INFO] Found {len(files)} frames in: {frames_dir}")

    H, W = size_hw
    tfm = transforms.Compose(
        [
            transforms.Resize((H, W)),
            transforms.ToTensor(),
        ]
    )

    frames = []
    for f in files:
        img = Image.open(f).convert("RGB")
        t = tfm(img)          # (3, H, W) in [0,1]
        t = t * 2.0 - 1.0     # -> [-1,1]
        frames.append(t)

    frames = torch.stack(frames, dim=0)       # (F, 3, H, W)
    frames = frames.unsqueeze(0).to(device)   # (1, F, 3, H, W)
    return frames


def run_single_prompt(
    *,
    prompt_key: str,
    prompt_text: str,
    device: torch.device,
    frame_rate: float,
    train_params: dict,
    model_cfg: dict,
    world_model,
    text_encoder,
    images: torch.Tensor,
    video_id: str,
    H: int,
    W: int,
):
    use_text = should_use_text(prompt_text)

    print("\n" + "-" * 90)
    print(f"[RUN] prompt_key   = {prompt_key}")
    print(f"[RUN] use_text     = {use_text}")
    print(f"[RUN] caption      = '{prompt_text}'")

    # ---------------- PREPARE OUTPUT FOLDER ----------------
    run_id = time.strftime("%Y%m%d-%H%M%S")

    # BASE_OUT_DIR/<prompt_key>/run_<timestamp>_<video_id>
    out_dir = Path(BASE_OUT_DIR) / prompt_key / f"run_{run_id}_{video_id}"
    context_dir = out_dir / "context_frames"
    gen_dir = out_dir / "gen_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # record caption for reproducibility
    (out_dir / "caption.txt").write_text(prompt_text + "\n", encoding="utf-8")

    print(f"[INFO] Saving results to: {out_dir}")

    # ---------------- ENCODE + ROLL OUT ----------------
    with torch.no_grad():
        text_emb = text_encoder([prompt_text]) if use_text else None

        latents = world_model.encode_frames(images)
        B, F_lat, E, H_lat, W_lat = latents.shape
        print(f"[INFO] Encoded latents shape: {latents.shape}")

        context = latents
        context_frames = context.shape[1]
        print(f"[INFO] Using {context_frames} context frames.")

        total_frames_needed = int(DESIRED_SECONDS * frame_rate)
        num_gen_frames = max(total_frames_needed - context_frames, 1)

        print(f"[INFO] Target clip length: {DESIRED_SECONDS}s")
        print(f"[INFO] => total_frames_needed = {total_frames_needed}")
        print(f"[INFO] => num_gen_frames to generate = {num_gen_frames}")

        rollout_kwargs = dict(
            x_0=context,
            num_gen_frames=num_gen_frames,
            latent_input=True,
            eta=0.0,
            NFE=30,
            sample_with_ema=True,
            num_samples=images.size(0),
        )
        if use_text:
            rollout_kwargs["text_emb"] = text_emb

        _, decoded = world_model.roll_out(**rollout_kwargs)

        print(f"[INFO] Generated frames shape: {decoded.shape}")

    # ---------------- SAVE CONTEXT + GENERATED FRAMES (PNGs) ----------------
    for t in range(images.shape[1]):
        frame = images[0, t]
        frame_vis = (frame + 1.0) / 2.0
        vutils.save_image(frame_vis, context_dir / f"context_{t:02d}.png")

    for t in range(decoded.shape[1]):
        frame = decoded[0, t]
        frame_vis = (frame + 1.0) / 2.0
        vutils.save_image(frame_vis, gen_dir / f"gen_{t:02d}.png")

    # ---------------- BUILD GIFs ----------------
    combined_frames = [images[0, t] for t in range(images.shape[1])] + [decoded[0, t] for t in range(decoded.shape[1])]
    combined_gif_frames = [to_uint8_image(frm) for frm in combined_frames]
    combined_gif_path = out_dir / "combined_context_plus_gen.gif"
    imageio.mimsave(combined_gif_path, combined_gif_frames, fps=int(frame_rate), loop=0)
    print(f"[GIF] Saved combined GIF: {combined_gif_path}")

    gen_only_frames = [decoded[0, t] for t in range(decoded.shape[1])]
    gen_only_gif_frames = [to_uint8_image(frm) for frm in gen_only_frames]
    gen_only_gif_path = out_dir / "generated_only.gif"
    imageio.mimsave(gen_only_gif_path, gen_only_gif_frames, fps=int(frame_rate), loop=0)
    print(f"[GIF] Saved generated-only GIF: {gen_only_gif_path}")

    print("[DONE] prompt_key:", prompt_key)
    print(f"       out_dir: {out_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] USE_FINETUNED_WEIGHTS = {USE_FINETUNED_WEIGHTS}")

    fm_cfg, model_cfg, train_params, val_params = load_fm_config()

    size = train_params["size"]
    H, W = int(size[0]), int(size[1])
    frame_rate = float(train_params.get("target_frame_rate", 5.0))

    print(f"[INFO] Using frame size: {H}x{W}")
    print(f"[INFO] Using target_frame_rate: {frame_rate} fps")

    images = load_frames_from_folder(FRAMES_DIR, (H, W), device)
    _, F, _, _, _ = images.shape
    video_id = Path(FRAMES_DIR).name
    print(f"\n[INFO] Frames folder name (video_id): {video_id}")
    print(f"[INFO] Number of context frames provided: {F}")

    # ---------------- BUILD MODELS ONCE ----------------
    any_prompt_uses_text = any(should_use_text(p) for p in TEXT_PROMPTS.values())
    text_encoder = create_text_encoder(device) if any_prompt_uses_text else None
    if not any_prompt_uses_text:
        print("[INFO] All prompts empty -> no text encoder will be created.")

    max_frames = train_params["num_frames"]
    world_model = create_world_model(model_cfg, max_frames, device, ORBIT_CKPT)
    world_model.eval()

    # ---------------- LOAD FINETUNED ADALN CHECKPOINT (ONCE) ----------------
    if USE_FINETUNED_WEIGHTS:
        print(f"\n[INFO] Loading finetuned AdaLN checkpoint from: {SAVE_PATH}")
        ckpt = torch.load(SAVE_PATH, map_location=device)
        state_dict = ckpt["state_dict"]

        inject_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INJECT][{inject_ts}] Injecting finetuned weights into world_model (strict=False)...")

        missing, unexpected = world_model.load_state_dict(state_dict, strict=False)

        inject_done_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INJECT][{inject_done_ts}] Injection complete for world_model.")
        print(f"[INFO] Loaded finetuned world model weights.")
        print(f"       Missing keys   : {len(missing)}")
        print(f"       Unexpected keys: {len(unexpected)}")

        if text_encoder is not None:
            text_proj_state = ckpt["text_proj"]
            proj_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INJECT][{proj_ts}] Injecting finetuned weights into text_encoder.proj (strict=True)...")
            text_encoder.proj.load_state_dict(text_proj_state, strict=True)
            proj_done_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[INJECT][{proj_done_ts}] Injection complete for text_encoder.proj.")
            print("[INFO] Loaded finetuned text projection weights.\n")
        else:
            print("[INFO] No text encoder instantiated; skipping text projection injection.\n")
    else:
        print("\n[INFO] USE_FINETUNED_WEIGHTS=False; using original ORBIT_CKPT weights (no injection).")

    # ---------------- RUN ALL PROMPTS ----------------
    for k, prompt in TEXT_PROMPTS.items():
        if should_use_text(prompt) and text_encoder is None:
            raise RuntimeError("Prompt requires text conditioning but text_encoder is None. This should not happen.")

        run_single_prompt(
            prompt_key=k,
            prompt_text=prompt,
            device=device,
            frame_rate=frame_rate,
            train_params=train_params,
            model_cfg=model_cfg,
            world_model=world_model,
            text_encoder=text_encoder,
            images=images,
            video_id=video_id,
            H=H,
            W=W,
        )

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
