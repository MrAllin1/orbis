import os
import torch
from tqdm import tqdm

# ==== PROJECT IMPORTS ====
from finetuning.text_encoder import CLIPTextEncoder
from networks.DiT.dit import STDiT
from data.covla_dataset import CoVLAOrbisMultiFrame
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ====== HYPERPARAMS ======
BATCH_SIZE = 2
LR = 1e-5
EPOCHS = 1
STEPS_PER_EPOCH = 200  # since dataset is streaming, we choose steps manually
CONTEXT_FRAMES = 4
TARGET_FRAMES = 4
DIFF_STEPS = 1000
# =========================

# ===================== Dataset (STREAMING) =====================
# NOTE: streaming=True => no len(), can't use standard DataLoader with RandomSampler
train_ds = CoVLAOrbisMultiFrame(
    split="train",
    num_frames=CONTEXT_FRAMES + TARGET_FRAMES,
    streaming=True,
    captions_dir="data/covla_captions",
)

def infinite_stream(dataset):
    """
    Creates an infinite generator over the streaming dataset.
    Each element is a dict with keys: 'images', 'caption', 'frame_rate'.
    """
    while True:
        for sample in dataset:
            yield sample

data_stream = infinite_stream(train_ds)
# ===============================================================

# ===================== Text Encoder =====================
text_encoder = CLIPTextEncoder(device=device).to(device)
for p in text_encoder.model.parameters():
    p.requires_grad = False
print("Text encoder initialized & frozen.")
# =======================================================

# ===================== Orbis Model (STDiT) =====================
# This matches your previous STDiT setup; checkpoint keys look like 'vit.pos_embed', etc.
model = STDiT(
    input_size=[16, 16],
    patch_size=2,
    in_channels=3,      # still working in RGB space (no tokenizer here)
    hidden_size=1152,
    depth=28,
    num_heads=16,
    mlp_ratio=4.0,
    max_num_frames=8,
    dropout=0.0,
).to(device)

ckpt_path = "logs_wm/orbis_288x512/checkpoints/last.ckpt"
print(f"Loading Orbis model weights from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

state_dict = ckpt.get("state_dict", ckpt)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Loaded checkpoint (strict=False).")
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))
# ===============================================================

# ===================== Optimizer =====================
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(text_encoder.proj.parameters()),
    lr=LR,
    weight_decay=0.01,
)
# ======================================================

step = 0
model.train()


def collate_batch(samples):
    """
    Manual collation because we're not using DataLoader.
    samples: list[dict] of length BATCH_SIZE
    """
    # images: list of (F, C, H, W) tensors -> (B, F, C, H, W)
    imgs = torch.stack([s["images"] for s in samples], dim=0)

    # captions: list of list[str]
    captions = [s["caption"] for s in samples]

    # frame_rate: scalar or tensor per sample -> (B,)
    # depending on how you stored it in CoVLAOrbisMultiFrame
    frame_rates = torch.tensor(
        [s["frame_rate"] for s in samples],
        dtype=torch.float32,
    )

    return imgs, captions, frame_rates


# ===================== Training Loop =====================
for epoch in range(EPOCHS):
    pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}")

    for _ in pbar:
        # ---- 1) Build batch manually from streaming dataset ----
        samples = [next(data_stream) for _ in range(BATCH_SIZE)]
        imgs, captions, frame_rates = collate_batch(samples)

        imgs = imgs.to(device)              # (B, F, C, H, W)
        frame_rates = frame_rates.to(device)

        # ---- 2) Basic sanity on frame count ----
        B, F, C, H, W = imgs.shape
        assert F == CONTEXT_FRAMES + TARGET_FRAMES, \
            f"Expected {CONTEXT_FRAMES + TARGET_FRAMES} frames, got {F}"

        context = imgs[:, :CONTEXT_FRAMES]      # (B, Ctx, C, H, W)
        target = imgs[:, CONTEXT_FRAMES:]       # (B, Tgt, C, H, W)

        # ---- 3) Clean up captions -> 1 string per video ----
        video_caps = []
        for cap_list in captions:
            non_empty = [c.strip() for c in cap_list if c.strip() != ""]
            video_caps.append(" ".join(non_empty) if non_empty else "no caption")

        text_emb = text_encoder(video_caps)  # (B, 1152) after projection

        # ---- 4) Diffusion step ----
        t = torch.randint(0, DIFF_STEPS, (B,), device=device)
        noise = torch.randn_like(target)
        noisy_target = target + noise

        pred = model(
            target=noisy_target,
            context=context,
            t=t,
            frame_rate=frame_rates,
            text_emb=text_emb,
        )

        loss = torch.mean((pred - noise) ** 2)

        # ---- 5) Backprop ----
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        pbar.set_postfix({"loss": float(loss)})

# ===================== Save Model =====================
save_path = "finetuned_orbis_text.ckpt"
torch.save(
    {
        "model": model.state_dict(),
        "text_proj": text_encoder.proj.state_dict(),
    },
    save_path,
)

print(f"\n🔥 FINISHED — saved: {save_path}")
