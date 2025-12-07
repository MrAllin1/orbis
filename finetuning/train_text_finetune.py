import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm
import yaml

from omegaconf import OmegaConf
from util import instantiate_from_config  # make sure this import path is correct

# ==== PROJECT IMPORTS ====
from finetuning.text_encoder import CLIPTextEncoder
from data.covla_dataset import CoVLAOrbisMultiFrame
from networks.DiT.dit import STDiT
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Separate device for tokenizer (CPU to save VRAM)
tokenizer_device = torch.device("cpu")

# ====== BASE PATHS ======
BASE_WORK_UNIS = "/work/dlclarge2/alidemaa-text-control-orbis/orbis"
BASE_WORK_KIT = "/home/fr/fr_fr/fr_aa533/work/orbis"

# ====== LOAD FM/YAML CONFIG (STAGE2, NOW COVLA) ======
FM_CONFIG_PATH = (
    "/work/dlclarge2/alidemaa-text-control-orbis/orbis/"
    "finetuning/stage2_baseline_covla_bev.yaml"
)
print(f"Loading FM config from: {FM_CONFIG_PATH}")
with open(FM_CONFIG_PATH, "r") as f:
    fm_cfg = yaml.safe_load(f)

model_cfg = fm_cfg["model"]["params"]

# ===================== DATA PARAMS FROM YAML (CoVLA) =====================
data_cfg = fm_cfg["data"]["params"]

train_entry = data_cfg["train"][0]
train_target = train_entry["target"]
train_params = dict(train_entry["params"])  # copy to allow setdefault

print(f"Train dataset target from YAML: {train_target}")

# Optional: fallback for paths if they are NOT present in YAML
CAPTIONS_DIR_DEFAULT = f"{BASE_WORK_UNIS}/data/covla_captions"
VIDEOS_DIR_DEFAULT   = f"{BASE_WORK_UNIS}/data/covla_100_videos"

train_params.setdefault("captions_dir", CAPTIONS_DIR_DEFAULT)
train_params.setdefault("videos_dir",   VIDEOS_DIR_DEFAULT)

YAML_SIZE        = train_params["size"]
YAML_NUM_FRAMES  = train_params["num_frames"]
YAML_STORED_RATE = train_params.get("stored_data_frame_rate", 20)
YAML_TARGET_RATE = train_params.get("target_frame_rate", 5)
YAML_SCALE_MIN   = train_params.get("scale_min", 0.75)
YAML_SCALE_MAX   = train_params.get("scale_max", 1.0)

print("YAML data params (train / CoVLA):")
print(f"  size            = {YAML_SIZE}")
print(f"  num_frames      = {YAML_NUM_FRAMES}")
print(f"  stored_rate     = {YAML_STORED_RATE}")
print(f"  target_rate     = {YAML_TARGET_RATE}")
print(f"  scale_min/max   = {YAML_SCALE_MIN}, {YAML_SCALE_MAX}")
print(f"  captions_dir    = {train_params['captions_dir']}")
print(f"  videos_dir      = {train_params['videos_dir']}")

# ====== OTHER PATHS ======
ORBIT_CKPT      = f"{BASE_WORK_UNIS}/logs_wm/orbis_288x512/checkpoints/last.ckpt"
SAVE_PATH       = f"{BASE_WORK_UNIS}/finetuning/finetuned_orbis_AdaLN.ckpt"
CHECKPOINT_PATH = f"{BASE_WORK_UNIS}/finetuning/checkpoints/adaln_debug.ckpt"  # separate debug ckpt

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

# ====== HYPERPARAMS (DEBUG) ======
BATCH_SIZE      = 1
LR              = 1e-4
EPOCHS          = 2          # DEBUG: only 2 epochs
STEPS_PER_EPOCH = 20         # DEBUG: fewer steps

CONTEXT_FRAMES  = 3
TARGET_FRAMES   = 3
MAX_FRAMES      = CONTEXT_FRAMES + TARGET_FRAMES  # should be 6

beta_start = 0.0001
beta_end   = 0.02
DIFF_STEPS = 1000
betas = torch.linspace(beta_start, beta_end, DIFF_STEPS, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# ===============================================================

# ===================== Dataset (CoVLA from YAML config) =====================
train_ds = CoVLAOrbisMultiFrame(**train_params)

def infinite_stream(ds):
    it = iter(ds)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(ds)
        except Exception:
            it = iter(ds)

data_stream = infinite_stream(train_ds)

# ===================== Text Encoder =================
text_encoder = CLIPTextEncoder(device=device).to(device)
text_encoder.eval()
for p in text_encoder.model.parameters():
    p.requires_grad = False
print("Text encoder initialized & frozen.")

# ===============================================================
#                 TOKENIZER: LOAD FROM STAGE1 CONFIG
# ===============================================================
print("\nLoading tokenizer from stage1 config...")

tk_cfg = model_cfg["tokenizer_config"]
tokenizer_folder = os.path.expandvars(tk_cfg["folder"])
TOKENIZER_CKPT = os.path.join(tokenizer_folder, tk_cfg["ckpt_path"])
tk_config_path = os.path.join(tokenizer_folder, "config.yaml")

tk_config = OmegaConf.load(tk_config_path)

# clean unsupported encoder kwargs
try:
    enc_params = tk_config["model"]["params"]["encoder_config"]["params"]
    if "use_pretrained_weights" in enc_params:
        enc_params.pop("use_pretrained_weights")
except Exception:
    pass

tokenizer = instantiate_from_config(tk_config["model"]).to(tokenizer_device)

tk = torch.load(TOKENIZER_CKPT, map_location="cpu")
tokenizer.load_state_dict(tk["state_dict"], strict=True)
tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad = False

print("Tokenizer ready.")

# ===================== STDiT (world model) =====================
print("\nLoading STDiT (world model)...")

gen_cfg = model_cfg["generator_config"]["params"]
GEN_INPUT_SIZE  = gen_cfg["input_size"]        # [H, W]
GEN_IN_CHANNELS = gen_cfg["in_channels"]
GEN_HIDDEN_SIZE = gen_cfg.get("hidden_size", 768)
GEN_DEPTH       = gen_cfg.get("depth", 12)
GEN_NUM_HEADS   = gen_cfg.get("num_heads", 12)
GEN_MLP_RATIO   = gen_cfg.get("mlp_ratio", 4)
GEN_DROPOUT     = gen_cfg.get("dropout", 0.0)
GEN_LEARN_SIGMA  = gen_cfg.get("learn_sigma", False)   # 🔴 ADD THIS

model = STDiT(
    input_size=GEN_INPUT_SIZE,
    patch_size=1,
    in_channels=GEN_IN_CHANNELS,
    hidden_size=GEN_HIDDEN_SIZE,
    depth=GEN_DEPTH,
    num_heads=GEN_NUM_HEADS,
    mlp_ratio=GEN_MLP_RATIO,
    max_num_frames=MAX_FRAMES,
    dropout=GEN_DROPOUT,
    learn_sigma=GEN_LEARN_SIGMA,
).to(device)

# DEBUG: check final_layer before loading pretrained weights
print("DEBUG: final_layer.linear mean abs BEFORE ckpt load:",
      model.final_layer.linear.weight.abs().mean().item())

stdit_ckpt = torch.load(ORBIT_CKPT, map_location="cpu")
model.load_state_dict(stdit_ckpt.get("state_dict", stdit_ckpt), strict=False)

# DEBUG: check final_layer after loading pretrained weights
print("DEBUG: final_layer.linear mean abs AFTER ckpt load:",
      model.final_layer.linear.weight.abs().mean().item())

print("STDiT pretrained weights loaded (strict=False).")

# ===================== FREEZE / UNFREEZE (DEBUG) =====================
# DEBUG: for this short run, unfreeze EVERYTHING in STDiT
# to check if the training pipeline actually reduces loss.
for name, p in model.named_parameters():
    p.requires_grad = True

# Still train text_encoder.proj as well
for p in text_encoder.proj.parameters():
    p.requires_grad = True

# DEBUG: log trainable params
num_train_params = 0
print("\nDEBUG: Trainable parameters in STDiT + text proj:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print("  TRAINABLE:", name, p.shape)
        num_train_params += p.numel()
print("Total trainable STDiT params:", num_train_params)

proj_params = sum(p.numel() for p in text_encoder.proj.parameters() if p.requires_grad)
print("Total trainable text_proj params:", proj_params)
print("Total trainable params (all):", num_train_params + proj_params)

train_params_list = list(model.parameters()) + list(text_encoder.proj.parameters())

optimizer = torch.optim.AdamW(train_params_list, lr=LR, weight_decay=0.01)

# =========================================================
#              CHECKPOINT LOAD / SAVE (DEBUG)
# =========================================================
def save_checkpoint(epoch, step):
    """Atomic save of training state."""
    tmp_path = CHECKPOINT_PATH + ".tmp"
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "text_proj_state_dict": text_encoder.proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, CHECKPOINT_PATH)
    print(f"[CKPT] Saved checkpoint at epoch={epoch}, step={step}", flush=True)

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"[CKPT] Found existing checkpoint: {CHECKPOINT_PATH}")
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        text_encoder.proj.load_state_dict(ckpt["text_proj_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        last_epoch = ckpt.get("epoch", 0)
        start_epoch = last_epoch + 1  # resume from next epoch
        print(f"[CKPT] Resuming from epoch={start_epoch}")
    except Exception as e:
        print(f"[CKPT] Failed to load checkpoint ({e}) → starting from scratch")
        start_epoch = 0
else:
    print("[CKPT] No existing checkpoint found → starting from scratch")

# ================ TOKENIZER LATENT ENCODING =================
def encode_latents(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, F, C, H, W) on CPU
    returns: (B, F, C_lat, H_lat, W_lat) on main device (GPU)
    """
    B, F, C, H, W = imgs.shape
    imgs = imgs.reshape(B * F, C, H, W).to(tokenizer_device)

    with torch.no_grad():
        encoded = tokenizer.encode(imgs)
        quantized = encoded["quantized"]

        if isinstance(quantized, tuple):
            q_rec, q_sem = quantized
            lat = torch.cat([q_rec, q_sem], dim=1)
        else:
            lat = quantized

    _, C_lat, H_lat, W_lat = lat.shape
    latents = lat.view(B, F, C_lat, H_lat, W_lat)
    latents = latents.to(device)
    return latents

# ===================== TRAIN (DEBUG) =====================
print("\nStarting AdaLN DEBUG fine-tuning (2 epochs, 20 steps)...\n")
model.train()

for epoch in range(start_epoch, EPOCHS):
    print(f"[TRAIN] Starting epoch {epoch + 1}/{EPOCHS}", flush=True)

    epoch_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        # 1) load one (video, caption) sample
        s = next(data_stream)
        imgs = s["images"].unsqueeze(0)  # (1, F, C, H, W)

        # 2) tokenizer → latents
        latents = encode_latents(imgs)
        context = latents[:, :CONTEXT_FRAMES]
        target  = latents[:, CONTEXT_FRAMES:]

        # 3) text encoder
        text = s["caption"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "no caption"
        text_emb = text_encoder([text]).to(device)

        # 4) DDPM noise
        t = torch.randint(0, DIFF_STEPS, (1,), device=device)
        alpha_bar = alphas_cumprod[t].view(1, 1, 1, 1, 1)
        noise = torch.randn_like(target)
        noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1 - alpha_bar) * noise

        frame_rate = torch.tensor([train_ds.target_rate], device=device)

        # 5) STDiT forward
        pred = model(noisy, context, t, frame_rate, text_emb=text_emb)

        # 6) loss + backward
        loss = torch.mean((pred - noise) ** 2)
        optimizer.zero_grad()
        loss.backward()

        # DEBUG: print grad stats on first step
        if epoch == start_epoch and step == 0:
            with torch.no_grad():
                print("DEBUG: loss on first step:", loss.item())
                # check one or two layer grads
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        print("DEBUG: first trainable grad mean abs:",
                              name, p.grad.abs().mean().item())
                        break

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    print(f"Epoch {epoch} finished — avg_loss={avg_loss:.4f}", flush=True)

    save_checkpoint(epoch, step)

# ================= SAVE FINAL (DEBUG) =================
torch.save(
    {
        "state_dict": model.state_dict(),
        "text_proj": text_encoder.proj.state_dict(),
    },
    SAVE_PATH,
)

print(f"\nDone! DEBUG AdaLN fine-tune saved → {SAVE_PATH}")
