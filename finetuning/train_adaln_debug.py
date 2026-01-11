# finetuning/train_adaln.py
import os
import time
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter  # <-- NEW

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
    create_tokenizer,      # kept for compatibility if needed elsewhere
    create_world_model,
)
from finetuning.adaln_utils import (
    infinite_stream,
    save_checkpoint,
    load_checkpoint_if_exists,
)
def safe_mean(t):
    return t.abs().mean().item() if t is not None else 0.0

# ================== DEVICES ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_device = torch.device("cpu")
print(f"Using device: {device}")

# ================== CONFIG / DATASET ==================
fm_cfg, model_cfg, train_params, val_params = load_fm_config()
hp = TrainHyperparams()

CONTEXT_FRAMES = hp.context_frames
TARGET_FRAMES = hp.target_frames
MAX_FRAMES = CONTEXT_FRAMES + TARGET_FRAMES  # kept for clarity

# ---- TRAIN DATASET + STREAM ----
train_ds = CoVLAOrbisMultiFrame(**train_params)
train_stream = infinite_stream(train_ds)

# ---- VALIDATION DATASET + STREAM (if provided) ----
if val_params is not None:
    val_ds = CoVLAOrbisMultiFrame(**val_params)
    val_stream = infinite_stream(val_ds)
else:
    val_ds = None
    val_stream = None
    print("[WARN] No validation config found; training without val loop.")

# ================== MODELS ==================
text_encoder = create_text_encoder(device)

# World model = FM ModelIF (encodes frames, adds noise, roll_out, etc.)
model = create_world_model(model_cfg, MAX_FRAMES, device, ORBIT_CKPT)

# ================== FREEZE / UNFREEZE PARAMS ==================
# 1) Freeze everything in the world model
for p in model.parameters():
    p.requires_grad = False

model.eval()  # keep tokenizer / global model in eval mode (we only train vit + text proj)

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

# Sanity check: collect trainable params (AdaLN + text proj)
train_params_list = [p for p in vit.parameters() if p.requires_grad] + \
                    list(text_encoder.proj.parameters())
assert len(train_params_list) > 0, "No trainable parameters found!"

optimizer = torch.optim.AdamW(train_params_list, lr=hp.lr, weight_decay=0.01)

# checkpoint dir
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
start_epoch = load_checkpoint_if_exists(
    CHECKPOINT_PATH, model, text_encoder, optimizer, device
)

# ================== RUN / SAMPLES & TENSORBOARD DIRS ==================
# One folder per training run (prefer SLURM_JOB_ID, fallback to timestamp)
RUN_ID = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))

SAMPLES_DIR = os.path.join("finetuning", "samples", RUN_ID)
os.makedirs(SAMPLES_DIR, exist_ok=True)
print(f"[RUN] Saving samples to: {SAMPLES_DIR}", flush=True)

LOG_DIR = os.path.join("finetuning", "runs", RUN_ID)  # for TensorBoard
os.makedirs(LOG_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=LOG_DIR)
print(f"[RUN] TensorBoard logs in: {LOG_DIR}", flush=True)

# global step for TensorBoard (counts train steps)
global_step = start_epoch * hp.steps_per_epoch


# ================== SAMPLING UTILS ==================
@torch.no_grad()
def sample_and_save(model, data_stream, step_label="init"):
    """
    Take one example, encode with model.encode_frames, roll out 1 frame
    WITH TEXT CONDITIONING, and save it.
    """
    s = next(data_stream)
    imgs = s["images"].unsqueeze(0).to(device)   # (1, F, C, H, W)

    # --- text conditioning from caption ---
    capt = s.get("caption", "")
    if isinstance(capt, list):
        text = capt[0].strip() if len(capt) > 0 and isinstance(capt[0], str) else ""
    elif isinstance(capt, str):
        text = capt.strip()
    else:
        text = ""
    if not text:
        text = "no caption"

    # CLIPTextEncoder already runs on `device`
    text_emb = text_encoder([text])  # (1, D)

    # --- encode frames to latents ---
    latents = model.encode_frames(imgs)          # (1, F, E, H, W)
    context = latents[:, :-1]                    # (1, F-1, E, H, W)

    # Roll-out with text conditioning; frame_rate handled inside sample()
    _, decoded = model.roll_out(
        x_0=context,
        num_gen_frames=1,
        latent_input=True,
        eta=0.0,
        NFE=30,
        sample_with_ema=True,
        num_samples=imgs.size(0),
        text_emb=text_emb,
    )

    out_path = os.path.join(SAMPLES_DIR, f"sample_{step_label}.png")
    vutils.save_image(decoded[0, 0], out_path, normalize=True, scale_each=True)
    print(f"[SAMPLE] Saved {out_path}", flush=True)


@torch.no_grad()
def run_validation(model, text_encoder, val_stream, epoch, num_steps):
    """
    Simple validation loop: same loss as training but:
    - no optimizer
    - averaged over `num_steps` batches
    """
    if val_stream is None:
        return None

    model.vit.eval()
    total_loss = 0.0

    for _ in range(num_steps):
        s = next(val_stream)
        imgs = s["images"].unsqueeze(0).to(device)  # (1, F, C, H, W)

        # encode frames
        x = model.encode_frames(imgs)  # (B, F, E, H, W)
        B, F, E, H, W = x.size()

        if F == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:, :-1].clone()
            target = x[:, -1]

        # caption → text_emb
        caption = s["caption"]
        if isinstance(caption, str) and caption.strip():
            text = caption.strip()
        else:
            text = "no caption"
        text_emb = text_encoder([text]).to(device)

        # noise and t
        t = torch.rand((x.shape[0],), device=device)
        target_t, noise = model.add_noise(target, t)
        target_t = target_t.unsqueeze(1)

        # frame_rate tensor
        frame_rate = torch.full((x.shape[0],), val_ds.target_rate, device=device)

        # forward through STDiT
        pred = model.vit(
            target_t,
            context,
            t,
            frame_rate=frame_rate,
            text_emb=text_emb,
        )  # (B, 1, E, H, W)

        target_v = model.A(t) * target + model.B(t) * noise
        loss = torch.mean((pred.squeeze(1) - target_v) ** 2)
        total_loss += loss.item()

    avg_val_loss = total_loss / num_steps
    print(f"[VAL] Epoch {epoch + 1} — avg_val_loss={avg_val_loss:.4f}", flush=True)

    # go back to train mode afterwards
    model.vit.train()
    return avg_val_loss


# ================== TRAIN LOOP ==================
print("\nStarting AdaLN fine-tuning using fm_model noise & encoder...\n")
model.vit.train()

# Sample before training starts (text-conditioned)
sample_and_save(model, train_stream, "before_train")

for epoch in range(start_epoch, hp.epochs):
    print(f"[TRAIN] Starting epoch {epoch + 1}/{hp.epochs}", flush=True)
    epoch_loss = 0.0

    for step in range(hp.steps_per_epoch):
        s = next(train_stream)
        imgs = s["images"].unsqueeze(0).to(device)  # (B=1, F, C, H, W)

        # ---- Encode frames to latents (uses tokenizer internally) ----
        with torch.no_grad():
            x = model.encode_frames(imgs)           # (B, F, E, H, W)

        B, F, E, H, W = x.size()

        if F == 1:
            context = None
            target = x.squeeze(1)
        else:
            context = x[:, :-1].clone()
            target  = x[:, -1]

        # ---- text encoder ----
        caption = s["caption"]
        text = caption.strip() if isinstance(caption, str) and caption.strip() else "no caption"
        text_emb = text_encoder([text]).to(device)  # (B, D)

        # ---- FM noise ----
        if global_step < 1000:
            t = torch.rand((B,), device=device)
        else:
            t = torch.rand((B,), device=device) ** 2

        target_t, noise = model.add_noise(target, t)
        target_t = target_t.unsqueeze(1)

        frame_rate = torch.full((B,), train_ds.target_rate, device=device)

        # ================== FORWARD (TEXT ON) ==================
        pred_text = model.vit(
            target_t,
            context,
            t,
            frame_rate=frame_rate,
            text_emb=text_emb,
        )

        # ================== FORWARD (TEXT OFF) ==================
        with torch.no_grad():
            pred_no_text = model.vit(
                target_t,
                context,
                t,
                frame_rate=frame_rate,
                text_emb=None,
            )

        # ---- FM velocity target ----
        target_v = model.A(t) * target + model.B(t) * noise

        loss = torch.mean((pred_text.squeeze(1) - target_v) ** 2)

        optimizer.zero_grad()
        loss.backward()

        # ================== GRAD LOGS ==================
        if global_step % 100 == 0:
            # text MLP gradient
            for name, p in model.vit.named_parameters():
                if "text_mlp" in name and p.grad is not None:
                    g = p.grad.norm().item()
                    print(f"[GRAD][text_mlp] {name} | {g:.3e}")
                    writer.add_scalar("grad/text_mlp", g, global_step)
                    break

            # AdaLN gradient
            for name, p in model.vit.named_parameters():
                if "adaLN_modulation" in name and p.grad is not None:
                    g = p.grad.norm().item()
                    writer.add_scalar("grad/adaLN", g, global_step)
                    break

        optimizer.step()

        # ================== WEIGHT LOGS ==================
        if global_step % 200 == 0:
            with torch.no_grad():
                for name, p in model.vit.named_parameters():
                    if "text_mlp" in name:
                        w = p.abs().mean().item()
                        writer.add_scalar("weights/text_mlp", w, global_step)
                        break

                for name, p in model.vit.named_parameters():
                    if "adaLN_modulation" in name:
                        w = p.abs().mean().item()
                        writer.add_scalar("weights/adaLN", w, global_step)
                        break

        # ================== TEXT EFFECT METRIC ==================
        with torch.no_grad():
            text_delta = torch.mean(
                (pred_text - pred_no_text).abs()
            ).item()
            writer.add_scalar("diagnostics/text_delta", text_delta, global_step)

        epoch_loss += loss.item()
        writer.add_scalar("train/loss_step", loss.item(), global_step)
        global_step += 1

    avg_loss = epoch_loss / hp.steps_per_epoch
    print(f"[TRAIN] Epoch {epoch + 1} finished — avg_loss={avg_loss:.4f}", flush=True)

    writer.add_scalar("train/loss_epoch", avg_loss, epoch + 1)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

    # ================== VALIDATION ==================
    if val_stream is not None:
        val_steps = getattr(hp, "val_steps_per_epoch", 50)
        val_loss = run_validation(
            model, text_encoder, val_stream, epoch, num_steps=val_steps
        )
        if val_loss is not None:
            writer.add_scalar("val/loss_epoch", val_loss, epoch + 1)

    # ================== CHECKPOINT + SAMPLE ==================
    save_checkpoint(CHECKPOINT_PATH, epoch, step, model, text_encoder, optimizer)
    sample_and_save(model, train_stream, f"epoch{epoch+1}")



# ================ SAVE FINAL =================
torch.save(
    {
        "state_dict": model.state_dict(),                # full FM model (vit + ae etc.)
        "text_proj": text_encoder.proj.state_dict(),     # text projection head
    },
    SAVE_PATH,
)
print(f"\nDone! AdaLN fine-tune saved → {SAVE_PATH}")

# Close TensorBoard writer
writer.close()
print("[RUN] TensorBoard writer closed.")
