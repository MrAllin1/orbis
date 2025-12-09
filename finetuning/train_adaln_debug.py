# finetuning/train_adaln.py
import os
import time
import torch
import torch.nn as nn
import torchvision.utils as vutils

from data.covla_dataset import CoVLAOrbisMultiFrame

from finetuning.adaln_config import (
    TrainHyperparams,
    load_fm_config,
    ORBIT_CKPT,
    SAVE_PATH,
    CHECKPOINT_PATH,
)
from finetuning.adaln_models import (
    create_text_encoder,
    create_tokenizer,      # kept for compatibility
    create_world_model,
)
from finetuning.adaln_utils import (
    infinite_stream,
    save_checkpoint,
    load_checkpoint_if_exists,
)

# ================== DEVICES ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_device = torch.device("cpu")
print(f"Using device: {device}")

# ================== CONFIG / DATASET ==================
fm_cfg, model_cfg, train_params = load_fm_config()
hp = TrainHyperparams()

CONTEXT_FRAMES = hp.context_frames
TARGET_FRAMES = hp.target_frames
MAX_FRAMES = CONTEXT_FRAMES + TARGET_FRAMES  # not strictly needed anymore, but kept

# dataset + stream
train_ds = CoVLAOrbisMultiFrame(**train_params)
data_stream = infinite_stream(train_ds)

# ================== MODELS ==================
text_encoder = create_text_encoder(device)

# World model = FM Model (encodes frames, adds noise, roll_out, etc.)
model = create_world_model(model_cfg, MAX_FRAMES, device, ORBIT_CKPT)

# ================== FREEZE / UNFREEZE PARAMS ==================
# 1) Freeze everything in the world model
for p in model.parameters():
    p.requires_grad = False

# 2) Unfreeze only AdaLN-related params + small text MLP in STDiT
vit = model.vit

#   a) text MLP that injects text condition
if hasattr(vit, "text_mlp"):
    for p in vit.text_mlp.parameters():
        p.requires_grad = True

#   b) AdaLN modulations inside blocks
if hasattr(vit, "blocks"):
    for block in vit.blocks:
        if hasattr(block, "adaLN_modulation"):
            for p in block.adaLN_modulation.parameters():
                p.requires_grad = True
        if hasattr(block, "adaLN_time_attn_modulation"):
            for p in block.adaLN_time_attn_modulation.parameters():
                p.requires_grad = True

#   c) AdaLN in the final layer
if hasattr(vit, "final_layer") and hasattr(vit.final_layer, "adaLN_modulation"):
    for p in vit.final_layer.adaLN_modulation.parameters():
        p.requires_grad = True

# 3) Text encoder: freeze everything, then unfreeze only proj
for p in text_encoder.parameters():
    p.requires_grad = False
for p in text_encoder.proj.parameters():
    p.requires_grad = True

# Sanity check: collect trainable params
train_params_list = [p for p in vit.parameters() if p.requires_grad] + \
                    list(text_encoder.proj.parameters())
assert len(train_params_list) > 0, "No trainable parameters found!"

optimizer = torch.optim.AdamW(train_params_list, lr=hp.lr, weight_decay=0.01)

# checkpoint dir
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
start_epoch = load_checkpoint_if_exists(
    CHECKPOINT_PATH, model, text_encoder, optimizer, device
)
# ================== RUN / SAMPLES DIR ==================
# One folder per training run (prefer SLURM_JOB_ID, fallback to timestamp)
RUN_ID = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
SAMPLES_DIR = os.path.join("finetuning", "samples", RUN_ID)
os.makedirs(SAMPLES_DIR, exist_ok=True)
print(f"[RUN] Saving samples to: {SAMPLES_DIR}", flush=True)


# ================== SAMPLING UTILS ==================
@torch.no_grad()
def sample_and_save(model, data_stream, step_label="init"):
    """
    Take one example, encode with model.encode_frames, roll out 1 frame and save it.
    Uses the same temporal layout as pretraining: context = all but last frame.
    """
    s = next(data_stream)
    imgs = s["images"].unsqueeze(0).to(device)  # (1, F, C, H, W)

    # Encode to latents using fm_model's tokenizer + enc_scale
    latents = model.encode_frames(imgs)         # (1, F, E, H, W)
    context = latents[:, :-1]                  # all but last frame

    # Roll out 1 new frame (latent_input=True, we already gave latents)
    _, decoded = model.roll_out(
        context,
        num_gen_frames=1,
        latent_input=True,
        num_samples=imgs.size(0),
    )  # decoded: (B, gen_frames, 3, H, W)

    # Save in run-specific folder
    out_path = os.path.join(SAMPLES_DIR, f"sample_{step_label}.png")
    vutils.save_image(decoded[0, 0], out_path, normalize=True, scale_each=True)
    print(f"[SAMPLE] Saved {out_path}", flush=True)


# ================== TRAIN LOOP ==================
print("\nStarting AdaLN fine-tuning using fm_model noise & encoder...\n")
model.train()

# Sample before training starts
sample_and_save(model, data_stream, "before_train")

for epoch in range(start_epoch, hp.epochs):
    print(f"[TRAIN] Starting epoch {epoch + 1}/{hp.epochs}", flush=True)
    epoch_loss = 0.0

    for step in range(hp.steps_per_epoch):
        s = next(data_stream)
        imgs = s["images"].unsqueeze(0).to(device)  # (1, F, C, H, W)

        # ---- Encode frames to latents using fm_model ----
        with torch.no_grad():
            x = model.encode_frames(imgs)           # (B, F, E, H, W)

        B, F, E, H, W = x.size()

        # ===== Match FM pretraining temporal layout =====
        if F == 1:
            # Degenerate case: no context, just target
            context = None
            target = x.squeeze(1)                   # (B, E, H, W)
        else:
            context = x[:, :-1].clone()             # (B, F-1, E, H, W)
            target  = x[:, -1]                      # (B, E, H, W)

        # text encoder
        text = s["caption"] if isinstance(s["caption"], str) and s["caption"].strip() else "no caption"
        text_emb = text_encoder([text]).to(device)  # (B, D)

        # ======== ADD NOISE using fm_model.Model.add_noise ========
        # t ~ U(0,1) per batch element (B)
        t = torch.rand((x.shape[0],), device=device)
        target_t, noise = model.add_noise(target, t)   # both (B, E, H, W)
        target_t = target_t.unsqueeze(1)               # (B, 1, E, H, W)

        # Frame rate tensor (1D, length B)
        frame_rate = torch.full((x.shape[0],), train_ds.target_rate, device=device)

        # ---- Forward through STDiT backbone (with text embedding) ----
        # vit expects (B, T_predict=1, C, H, W), context, t, frame_rate, text_emb
        pred = model.vit(
            target_t,
            context,
            t,
            frame_rate=frame_rate,
            text_emb=text_emb,
        )  # (B, 1, E, H, W)

        # ===== FM-style velocity target (same as fm_model.training_step) =====
        target_v = model.A(t) * target + model.B(t) * noise   # (B, E, H, W)

        loss = torch.mean((pred.squeeze(1) - target_v) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if epoch == start_epoch and step == 0:
            print("DEBUG: first step loss =", loss.item())

    avg_loss = epoch_loss / hp.steps_per_epoch
    print(f"Epoch {epoch + 1} finished — avg_loss={avg_loss:.4f}", flush=True)
    save_checkpoint(CHECKPOINT_PATH, epoch, step, model, text_encoder, optimizer)

    # sample after each epoch
    sample_and_save(model, data_stream, f"epoch{epoch+1}")

# ================ SAVE FINAL =================
torch.save(
    {
        "state_dict": model.state_dict(),                # full FM model (vit + ae etc.)
        "text_proj": text_encoder.proj.state_dict(),
    },
    SAVE_PATH,
)
print(f"\nDone! AdaLN fine-tune saved → {SAVE_PATH}")
