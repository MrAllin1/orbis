import os
import time
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

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
    create_world_model,
)
from finetuning.adaln_utils import (
    infinite_stream,
    save_checkpoint,
    load_checkpoint_if_exists,
)

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ============================================================
# CONFIG / DATA
# ============================================================
fm_cfg, model_cfg, train_params, val_params = load_fm_config()
hp = TrainHyperparams()

CONTEXT_FRAMES = hp.context_frames
TARGET_FRAMES = hp.target_frames
MAX_FRAMES = CONTEXT_FRAMES + TARGET_FRAMES

train_ds = CoVLAOrbisMultiFrame(**train_params)
train_stream = infinite_stream(train_ds)

# ============================================================
# MODELS
# ============================================================
print("[INFO] Creating text encoder (FULLY TRAINABLE)")
text_encoder = create_text_encoder(device)
text_encoder.train()

# unfreeze entire CLIP + proj
for p in text_encoder.parameters():
    p.requires_grad = True

print("[INFO] Creating world model")
model = create_world_model(model_cfg, MAX_FRAMES, device, ORBIT_CKPT)
model.train()

# ============================================================
# FREEZE TOKENIZER (AE) — IMPORTANT
# ============================================================
print("[INFO] Freezing tokenizer / AE")
for p in model.ae.parameters():
    p.requires_grad = False
model.ae.eval()

# ============================================================
# UNFREEZE ENTIRE STDiT BACKBONE
# ============================================================
for p in model.vit.parameters():
    p.requires_grad = True

# Sanity
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] Trainable parameters (no tokenizer): {trainable / 1e6:.2f}M")

# ============================================================
# OPTIMIZER (NO TOKENIZER PARAMS)
# ============================================================
optimizer = torch.optim.AdamW(
    list(model.vit.parameters()) +
    list(text_encoder.parameters()),
    lr=hp.lr,
    weight_decay=0.01,
)

# ============================================================
# CHECKPOINT
# ============================================================
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
start_epoch = load_checkpoint_if_exists(
    CHECKPOINT_PATH, model, text_encoder, optimizer, device
)

# ============================================================
# LOGGING
# ============================================================
RUN_ID = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
LOG_DIR = os.path.join("finetuning", "runs", f"fully_unfrozen_{RUN_ID}")
SAMPLES_DIR = os.path.join("finetuning", "samples", f"fully_unfrozen_{RUN_ID}")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

writer = SummaryWriter(LOG_DIR)
global_step = start_epoch * hp.steps_per_epoch

# ============================================================
# SAMPLING
# ============================================================
@torch.no_grad()
def sample_and_save(step_label):
    s = next(train_stream)
    imgs = s["images"].unsqueeze(0).to(device)

    text = s["caption"] if isinstance(s["caption"], str) and s["caption"] else "no caption"
    text_emb = text_encoder([text])

    latents = model.encode_frames(imgs)
    context = latents[:, :-1]

    _, decoded = model.roll_out(
        x_0=context,
        num_gen_frames=1,
        latent_input=True,
        text_emb=text_emb,
        NFE=20,
        eta=0.0,
        num_samples=1,
    )

    out = os.path.join(SAMPLES_DIR, f"sample_{step_label}.png")
    vutils.save_image(decoded[0, 0], out, normalize=True)
    print(f"[SAMPLE] {out}")

# ============================================================
# TRAIN LOOP
# ============================================================
print("\n[TRAIN] Starting FULLY UNFROZEN (tokenizer frozen)\n")
sample_and_save("before_train")

for epoch in range(start_epoch, hp.epochs):
    epoch_loss = 0.0
    print(f"[TRAIN] Epoch {epoch + 1}/{hp.epochs}")

    for step in range(hp.steps_per_epoch):
        s = next(train_stream)
        imgs = s["images"].unsqueeze(0).to(device)

        # ---- Encode (AE frozen, but still forward) ----
        with torch.no_grad():
            x = model.encode_frames(imgs)

        B, F, E, H, W = x.shape

        if F == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:, :-1]
            target = x[:, -1]

        text = s["caption"] if isinstance(s["caption"], str) and s["caption"] else "no caption"
        text_emb = text_encoder([text])

        t = torch.rand((B,), device=device)
        target_t, noise = model.add_noise(target, t)
        target_t = target_t.unsqueeze(1)

        frame_rate = torch.full((B,), train_ds.target_rate, device=device)

        pred = model.vit(
            target_t,
            context,
            t,
            frame_rate=frame_rate,
            text_emb=text_emb,
        )

        target_v = model.A(t) * target + model.B(t) * noise
        loss = torch.mean((pred.squeeze(1) - target_v) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

    avg_loss = epoch_loss / hp.steps_per_epoch
    print(f"[TRAIN] Epoch {epoch + 1} avg_loss={avg_loss:.4f}")
    writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)

    save_checkpoint(
        CHECKPOINT_PATH,
        epoch,
        step,
        model,
        text_encoder,
        optimizer,
    )

    sample_and_save(f"epoch{epoch+1}")

# ============================================================
# SAVE FINAL
# ============================================================
torch.save(
    {
        "state_dict": model.state_dict(),
        "text_encoder": text_encoder.state_dict(),
    },
    SAVE_PATH.replace(".ckpt", "_fully_unfrozen_no_tokenizer.ckpt"),
)

writer.close()
print("\n[DONE] Fully unfrozen training finished (tokenizer frozen).")
