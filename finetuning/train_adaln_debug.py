# finetuning/train_adaln_debug.py
import os
import time

import torch
import torch.nn as nn

from data.covla_dataset import CoVLAOrbisMultiFrame

from finetuning.adaln_config import (
    TrainHyperparams,
    load_fm_config,
    make_diffusion_schedule,
    ORBIT_CKPT,
    SAVE_PATH,
    CHECKPOINT_PATH,
)
from finetuning.adaln_models import (
    create_text_encoder,
    create_tokenizer,
    create_world_model,
)
from finetuning.adaln_utils import (
    infinite_stream,
    encode_latents,
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
TARGET_FRAMES  = hp.target_frames
MAX_FRAMES     = CONTEXT_FRAMES + TARGET_FRAMES

# diffusion schedule
_, _, alphas_cumprod = make_diffusion_schedule(
    hp.beta_start, hp.beta_end, hp.diffusion_steps, device=device
)

# dataset + stream
train_ds = CoVLAOrbisMultiFrame(**train_params)
data_stream = infinite_stream(train_ds)

# ================== MODELS ==================
text_encoder = create_text_encoder(device)
tokenizer = create_tokenizer(model_cfg, tokenizer_device)
model = create_world_model(model_cfg, MAX_FRAMES, device, ORBIT_CKPT)

# unfreeze everything in STDiT + text proj (debug)
for p in model.parameters():
    p.requires_grad = True
for p in text_encoder.proj.parameters():
    p.requires_grad = True

train_params_list = list(model.parameters()) + list(text_encoder.proj.parameters())
optimizer = torch.optim.AdamW(train_params_list, lr=hp.lr, weight_decay=0.01)

# checkpoint dir
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
start_epoch = load_checkpoint_if_exists(
    CHECKPOINT_PATH, model, text_encoder, optimizer, device
)

# ================== TRAIN LOOP ==================
print("\nStarting AdaLN DEBUG fine-tuning...\n")
model.train()

for epoch in range(start_epoch, hp.epochs):
    print(f"[TRAIN] Starting epoch {epoch + 1}/{hp.epochs}", flush=True)
    epoch_loss = 0.0

    for step in range(hp.steps_per_epoch):
        # 1) load one (video, caption) sample
        s = next(data_stream)
        imgs = s["images"].unsqueeze(0)  
        
        # 2) tokenizer → latents
        latents = encode_latents(imgs, tokenizer, tokenizer_device, device)
        context = latents[:, :CONTEXT_FRAMES]
        target  = latents[:, CONTEXT_FRAMES:]

        # 3) text encoder
        text = s["caption"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "no caption"
        text_emb = text_encoder([text]).to(device)

        # 4) DDPM noise
        t = torch.randint(0, hp.diffusion_steps, (1,), device=device)
        alpha_bar = alphas_cumprod[t].view(1, 1, 1, 1, 1)
        noise = torch.randn_like(target)
        noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1 - alpha_bar) * noise

        frame_rate = torch.tensor([train_ds.target_rate], device=device)

        # 5) STDiT forwar
        pred = model(noisy, context, t, frame_rate, text_emb=text_emb)

        # 6) loss + backward
        loss = torch.mean((pred - noise) ** 2)
        optimizer.zero_grad()
        loss.backward()

        if epoch == start_epoch and step == 0:
            with torch.no_grad():
                print("DEBUG: loss on first step:", loss.item())
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        print("DEBUG: first trainable grad mean abs:",
                              name, p.grad.abs().mean().item())
                        break

        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / hp.steps_per_epoch
    print(f"Epoch {epoch} finished — avg_loss={avg_loss:.4f}", flush=True)
    save_checkpoint(CHECKPOINT_PATH, epoch, step, model, text_encoder, optimizer)

# ================ SAVE FINAL =================
torch.save(
    {
        "state_dict": model.state_dict(),
        "text_proj": text_encoder.proj.state_dict(),
    },
    SAVE_PATH,
)
print(f"\nDone! DEBUG AdaLN fine-tune saved → {SAVE_PATH}")
