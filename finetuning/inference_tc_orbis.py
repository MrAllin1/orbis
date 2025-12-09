import torch
import torchvision.transforms as T
from PIL import Image
import os
import sys
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image

from finetuning.adaln_models import create_text_encoder, create_world_model
from finetuning.adaln_config import load_fm_config, ORBIT_CKPT

# ================================================================
# SETUP
# ================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Load YAML config (fm_cfg = full yaml, model_cfg = model params, train_params = dataset settings)
print("[INFO] Loading configs from YAML...")
fm_cfg, model_cfg, train_params = load_fm_config()

# Extract resolution + num_frames from YAML
img_h, img_w = train_params["size"]        # e.g., [192, 336]
num_input_frames = train_params["num_frames"]    # e.g., 6

print(f"[INFO] YAML → Resolution = {img_h}x{img_w}, num_input_frames = {num_input_frames}")

# ================================================================
# CREATE MODELS
# ================================================================
print("[INFO] Creating text encoder...")
text_encoder = create_text_encoder(device)

print("[INFO] Creating world model...")
model = create_world_model(model_cfg, max_frames=num_input_frames, device=device, orbit_ckpt_path=ORBIT_CKPT)

# Load finetuned AdaLN weights
ckpt_path = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/finetuned_orbis_AdaLN.ckpt"
print(f"[INFO] Loading fine-tuned checkpoint: {ckpt_path}")
state = torch.load(ckpt_path, map_location=device)

model.load_state_dict(state["state_dict"], strict=False)
text_encoder.proj.load_state_dict(state["text_proj"], strict=False)

model.eval()
print("[INFO] Checkpoint loaded and model set to eval mode.")

# ================================================================
# LOAD FRAMES (N = num_input_frames from YAML)
# ================================================================
frame_dir = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/example"
print(f"[INFO] Loading exactly {num_input_frames} frames from: {frame_dir}")

frame_paths = sorted([
    os.path.join(frame_dir, f)
    for f in os.listdir(frame_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:num_input_frames]

assert len(frame_paths) == num_input_frames, (
    f"[ERROR] Expected {num_input_frames} frames based on YAML, but found {len(frame_paths)}"
)

# Resize to the exact YAML resolution
transform = T.Compose([
    T.Resize((img_h, img_w)),
    T.ToTensor(),
])

frames = []
for p in frame_paths:
    print(f"[INFO] Reading frame: {p}")
    frames.append(transform(Image.open(p).convert("RGB")))

frames = torch.stack(frames, dim=0)              # (F, 3, H, W)
frames = frames * 2 - 1                           # Normalize to [-1, 1]
frames = frames.unsqueeze(0).to(device)           # (1, F, 3, H, W)

print(f"[INFO] Frames tensor shape: {frames.shape}")

# ================================================================
# TEXT PROMPT
# ================================================================
prompt = "A car turning right at an intersection"
print(f"[INFO] Prompt: '{prompt}'")
text_emb = text_encoder([prompt]).to(device)

# ================================================================
# ENCODE → ROLL OUT
# ================================================================
num_gen_frames = 64
print(f"[INFO] Encoding and rolling out {num_gen_frames} frames...")
with torch.no_grad():
    latents = model.encode_frames(frames)
    print(f"[INFO] Latents shape: {latents.shape}")

    _, decoded = model.roll_out(
        x_0=latents,
        num_gen_frames=num_gen_frames,
        latent_input=True,
        num_samples=1,
        text_emb=text_emb,
    )  # decoded: (1, 64, 3, H, W)

# ================================================================
# CONVERT TO GIF (no intermediate PNGs)
# ================================================================
frames_list = [
    to_pil_image(decoded[0, i].clamp(-1, 1) * 0.5 + 0.5) for i in range(num_gen_frames)
]

gif_path = "generated_clip.gif"
frames_list[0].save(
    gif_path,
    save_all=True,
    append_images=frames_list[1:],
    duration=100,  # ms per frame
    loop=0,
)

print(f"[INFO] Saved animated GIF to: {gif_path}")
