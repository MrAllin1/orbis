import os
import sys
import torch
import yaml
from omegaconf import OmegaConf
import clip
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torchvision.transforms.functional as TF  # Add near top if not already imported

from util import instantiate_from_config

# === Adjust these paths for your setup ===
STAGE2_YAML = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/stage2_baseline_covla_bev.yaml"   # YAML config used for fine-tuning (stage 2)
FINETUNED_CKPT = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/finetuned_orbis_AdaLN.ckpt"    # The fine-tuned AdaLN checkpoint
CONTEXT_IMAGE_DIR = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/example"            # Directory with context frame images (e.g., 3 initial frames)
OUTPUT_DIR = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/output"                    # Directory to save the output predicted frames
TEXT_PROMPT = "The car approaches an intersection and turns left."  # Example text prompt for control

# === Device setup ===  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load fine-tuning config (stage 2 YAML) ===
with open(STAGE2_YAML, "r") as f:
    config = yaml.safe_load(f)
model_cfg = config["model"]["params"]
data_cfg = config.get("data", {}).get("params", {})

# === Load Tokenizer (VQ model) from its config ===
tk_cfg = model_cfg["tokenizer_config"]
tokenizer_folder = os.path.expandvars(tk_cfg["folder"])
tokenizer_ckpt_path = os.path.join(tokenizer_folder, tk_cfg["ckpt_path"])
tokenizer_config_path = os.path.join(tokenizer_folder, "config.yaml")
print(f"Loading tokenizer from folder: {tokenizer_folder}")
tk_config = OmegaConf.load(tokenizer_config_path)

# If the tokenizer config contains unsupported keys (like use_pretrained_weights), remove them
try:
    enc_params = tk_config["model"]["params"]["encoder_config"]["params"]
    if "use_pretrained_weights" in enc_params:
        enc_params.pop("use_pretrained_weights", None)
except Exception as e:
    print("Warning: could not adjust tokenizer config:", e)

tokenizer = instantiate_from_config(tk_config["model"])
tokenizer.load_state_dict(torch.load(tokenizer_ckpt_path, map_location="cpu")["state_dict"], strict=True)
tokenizer = tokenizer.to(device)
tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad = False
print("Tokenizer loaded and ready.")

# === Define CLIP text encoder with projection (same as in training) ===
class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, model_name="ViT-B/32", device=device):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)  # loads CLIP model
        self.model.eval()
        # Projection layer (512 -> 768) to match Orbis hidden size
        self.proj = torch.nn.Linear(512, model_cfg["generator_config"]["params"].get("hidden_size", 768))
        # (Weights will be loaded from checkpoint, so init values don't matter)
    def forward(self, captions):
        tokens = clip.tokenize(captions).to(device)
        with torch.no_grad():
            clip_emb = self.model.encode_text(tokens)  # (B, 512) CLIP text embeddings
        clip_emb = clip_emb.to(self.proj.weight.dtype)
        return self.proj(clip_emb)  # project to (B, hidden_size)

# Instantiate CLIP text encoder and load fine-tuned projection weights
text_encoder = CLIPTextEncoder().to(device)
text_encoder.model.eval()  # CLIP model stays in eval/frozen
# === Load fine-tuned model checkpoint (AdaLN) ===
ckpt = torch.load(FINETUNED_CKPT, map_location=device)
if "text_proj" in ckpt:
    text_encoder.proj.load_state_dict(ckpt["text_proj"], strict=True)
    print("Loaded fine-tuned text projection weights.")
else:
    print("Warning: 'text_proj' not found in checkpoint!")

# === Define the STDiT model architecture (must match training config) ===
from networks.DiT.dit import STDiT  # Ensure this import path is correct or adjust
gen_params = model_cfg["generator_config"]["params"]
# Extract key parameters for STDiT
INPUT_SIZE  = gen_params["input_size"]        # e.g., [12, 21] for latent spatial size
PATCH_SIZE  = gen_params.get("patch_size", 1) # likely 1
IN_CHANNELS = gen_params["in_channels"]       # e.g., 16 (latent channels per codebook * 2 if using factorized latents)
HIDDEN_SIZE = gen_params.get("hidden_size", 768)
DEPTH       = gen_params.get("depth", 12)
NUM_HEADS   = gen_params.get("num_heads", 12)
MLP_RATIO   = gen_params.get("mlp_ratio", 4)
DROPOUT     = gen_params.get("dropout", 0.0)
MAX_FRAMES  = gen_params.get("max_num_frames", 6)  # max context+target frames the model was trained on

# Initialize model and load weights
model = STDiT(
    input_size=INPUT_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IN_CHANNELS,
    hidden_size=HIDDEN_SIZE,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    max_num_frames=MAX_FRAMES,
    dropout=DROPOUT,
).to(device)
model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
model.eval()
print("Loaded fine-tuned STDiT model.")

# === Prepare context frames ===
# Load context images from directory (sorted by filename)
context_files = sorted([f for f in os.listdir(CONTEXT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if len(context_files) == 0:
    raise FileNotFoundError(f"No images found in {CONTEXT_IMAGE_DIR}")
# Limit context frames to what the model can accept (MAX_FRAMES - at least 1 for target)
max_context = MAX_FRAMES - 1  # leave at least 1 frame for prediction
if len(context_files) > max_context:
    context_files = context_files[:max_context]
print(f"Using {len(context_files)} context frames: {context_files}")

# Load images, resize to training size, and convert to tensor
context_imgs = []
if "size" in data_cfg:
    target_H, target_W = data_cfg["size"]
else:
    raise ValueError("Training resolution 'size' missing in YAML data config.")

print(f"Expected resolution: {target_W}x{target_H}")
for fname in context_files:
    img = Image.open(os.path.join(CONTEXT_IMAGE_DIR, fname)).convert("RGB")
    img = TF.resize(img, (target_H, target_W), antialias=True)  # Resize before tensor conversion
    img_tensor = TF.to_tensor(img)  # Converts to (C, H, W), float in [0,1]
    context_imgs.append(img_tensor)

context_tensor = torch.stack(context_imgs, dim=0)  # shape: (F, C, H, W)

# === Encode context frames into latents ===
context_tensor = context_tensor.to(device)
with torch.no_grad():
    # The tokenizer might have an encode method similar to VQ-VAEs
    encoded = tokenizer.encode(context_tensor)
    # tokenizer.encode likely returns a dict with 'quantized' or 'latent'
    if isinstance(encoded, dict):
        if "quantized" in encoded:
            quantized = encoded["quantized"]
        elif "latents" in encoded:
            quantized = encoded["latents"]
        else:
            raise ValueError("Unknown tokenizer output format.")
    else:
        # If encode returns tensor directly
        quantized = encoded
    # If the tokenizer uses a two-stream output (e.g., tuple of rec and sem latents):
    if isinstance(quantized, tuple):
        q_rec, q_sem = quantized
        latents = torch.cat([q_rec, q_sem], dim=1)  # concatenate on channel dimension
    else:
        latents = quantized
    # Reshape back to (B, F, C_lat, H_lat, W_lat)
    B = 1  # batch size
    F = len(context_files)  # number of context frames you actually used
    C_lat = latents.shape[1]
    H_lat = latents.shape[2]
    W_lat = latents.shape[3]
    context_latents = latents.view(B, F, C_lat, H_lat, W_lat).to(device)

print(f"Context latents shape: {tuple(context_latents.shape)}")
original_context_latents = context_latents.detach().cpu()

# === Get text conditioning embedding ===
text_emb = text_encoder([TEXT_PROMPT])  # shape: (1, hidden_size)
print(f"Text prompt: \"{TEXT_PROMPT}\"")
text_emb = text_emb.to(device)

# === Set diffusion parameters (same as training) ===
DIFF_STEPS = model_cfg.get("diffusion_steps", 1000) if model_cfg.get("diffusion_steps") else 1000
beta_start = model_cfg.get("beta_start", 0.0001)
beta_end = model_cfg.get("beta_end", 0.02)
betas = torch.linspace(beta_start, beta_end, DIFF_STEPS, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # shape: (DIFF_STEPS,)
sqrt_alpha_cum = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alpha_cum = torch.sqrt(1 - alphas_cumprod)

# === Helper: one diffusion sampling for a given number of future frames ===
def generate_future_frames(context_latents, text_emb, num_future=3):
    """
    Generate `num_future` future frames latents given context_latents and text_emb.
    Uses diffusion sampling from noise to predicted latents.
    Returns a tensor of shape (1, num_future, C_lat, H_lat, W_lat).
    """
    model.eval()
    # Get frame_rate from data (if available) or default (like 5 Hz in training)
    frame_rate_val = data_cfg.get("target_frame_rate", 5)
    frame_rate = torch.tensor([frame_rate_val], device=device)
    B, f_context, C_lat, H_lat, W_lat = context_latents.shape
    # Initialize target latents with pure noise at t = T (last diffusion step)
    f_target = num_future
    # Shape for target latents noise: (B, f_target, C_lat, H_lat, W_lat)
    target_latents = torch.randn((B, f_target, C_lat, H_lat, W_lat), device=device)
    # Diffusion sampling loop (reverse process)
    for t in range(DIFF_STEPS - 1, -1, -1):  # t from T-1 down to 0
        # Prepare time step as tensor
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)
        # Model expects current noisy target and context
        with torch.no_grad():
            # The model forward signature: model(target, context, t, frame_rate, text_emb)
            # We provide current target estimate (target_latents) and context
            noise_pred = model(target_latents, context_latents, t_tensor, frame_rate, text_emb=text_emb)
        # If on last step (t==0), we will directly use x0 prediction (no noise added)
        if t == 0:
            # At t=0, no further noise to add; break after updating x to x0_pred.
            x0_pred = (target_latents - sqrt_one_minus_alpha_cum[t] * noise_pred) / sqrt_alpha_cum[t]
            target_latents = x0_pred
            break
        # Otherwise, compute x0 estimate from current prediction
        x0_pred = (target_latents - sqrt_one_minus_alpha_cum[t] * noise_pred) / sqrt_alpha_cum[t]
        # Option 1: **Deterministic (DDIM-like)**: use model's predicted noise for backward step
        # (This avoids random variations and yields a consistent result for given input)
        eps = noise_pred  
        # Option 2 (for diversity): sample new noise at each step: eps = torch.randn_like(target_latents)
        # Compute x_{t-1} using the DDIM update rule
        target_latents = sqrt_alpha_cum[t-1] * x0_pred + sqrt_one_minus_alpha_cum[t-1] * eps
    # After loop, target_latents should be approximate model output at t=0
    return target_latents

# === Generate future frames in a sliding-window fashion if needed ===
num_future_total = 3  # Number of future frames to generate in total (adjustable)
if num_future_total < 1:
    raise ValueError("num_future_total must be >= 1")
generated_latents_list = []
frames_to_generate = num_future_total

# We will generate in chunks up to model's capacity (MAX_FRAMES - current context length)
while frames_to_generate > 0:
    # Determine how many to generate in this iteration
    current_context_len = context_latents.shape[1]
    max_target_allowed = MAX_FRAMES - current_context_len
    # If model allows no target frames (shouldn't happen unless context==MAX_FRAMES)
    if max_target_allowed < 1:
        raise RuntimeError("Context frames count is too high; no room for target frames in model.")
    # Generate either the remainder needed or max allowed, whichever is smaller
    num_target = min(frames_to_generate, max_target_allowed)
    print(f"Generating {num_target} future frames (context length {current_context_len})...")
    new_target_latents = generate_future_frames(context_latents, text_emb, num_future=num_target)
    # Append the newly generated latents to list (detach to free computation graph)
    generated_latents_list.append(new_target_latents.detach().cpu())
    # Update frames_to_generate
    frames_to_generate -= num_target
    if frames_to_generate <= 0:
        break
    # Prepare context for next iteration:
    # We take the last `current_context_len` frames from the **combined** sequence of context + new_target
    # or simply the entire new_target if its length equals context length, depending on strategy.
    # Here, use *sliding window*: context for next step = last `f_context` frames of (old context + new_target).
    total_sequence = torch.cat([context_latents.cpu(), new_target_latents.cpu()], dim=1)
    # Determine new context length for next iteration (we keep same number of context frames as initial if possible)
    new_context_len = current_context_len
    if total_sequence.shape[1] < new_context_len:
        new_context_len = total_sequence.shape[1]
    context_latents = total_sequence[:, -new_context_len:, ...].to(device)
    print(f"New context prepared with {context_latents.shape[1]} frames for next iteration.")

# Concatenate original context + all generated frames for output
if len(generated_latents_list) > 0:
    all_generated_latents = torch.cat(generated_latents_list, dim=1)  # (1, total_gen, C, H, W)
    all_frames_latents = torch.cat(
        [original_context_latents, all_generated_latents], dim=1
    )  # (1, F_context + total_gen, C, H, W)
else:
    all_generated_latents = None
    all_frames_latents = original_context_latents

print(f"Total frames (including context + generated): {all_frames_latents.shape[1]}")

# === Decode latents back to images using the tokenizer ===
all_frames_latents = all_frames_latents.to(device)
with torch.no_grad():
    # Reshape to (B*frames, C_lat, H_lat, W_lat) for decoding
    B, total_frames, C_lat, H_lat, W_lat = all_frames_latents.shape
    latents_reshaped = all_frames_latents.view(B * total_frames, C_lat, H_lat, W_lat)

    # Try tokenizer.decode, fall back to forward
    try:
        recon = tokenizer.decode(latents_reshaped)
    except AttributeError:
        recon = tokenizer(latents_reshaped)

    # --- NEW: extract the actual reconstruction tensor from dict/tuple/list ---
    def extract_recon(x):
        # Case 1: dict-like output
        if isinstance(x, dict):
            for k in ["decoded", "recon", "x_recon", "x"]:
                if k in x:
                    return x[k]
            # Fallback: take first value
            return next(iter(x.values()))

        # Case 2: tuple/list -> almost always (recon, ...)
        if isinstance(x, (tuple, list)):
            # Assume first element is the reconstruction tensor
            return extract_recon(x[0])

        # Case 3: already a Tensor
        return x

    recon_img = extract_recon(recon)  # should now be a tensor [B*T, C, H, W]
    if not torch.is_tensor(recon_img):
        raise TypeError(f"Tokenizer decode returned non-tensor even after extraction: {type(recon_img)}")

    # If tokenizer outputs in [-1, 1], map to [0, 1]; if already [0, 1], this is harmless
    if recon_img.min().item() < 0.0:
        recon_img = (recon_img + 1.0) / 2.0

    recon_img = recon_img.clamp(0.0, 1.0)
    recon_img = recon_img.view(B, total_frames, 3, recon_img.shape[-2], recon_img.shape[-1])
    recon_img = recon_img.cpu()


# === Save output frames as images ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
for i in range(total_frames):
    frame = recon_img[0, i]
    frame_img = Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    frame_img.save(os.path.join(OUTPUT_DIR, f"frame_{i:03d}.png"))
print(f"Saved {total_frames} frames (context + generated) to {OUTPUT_DIR}")
