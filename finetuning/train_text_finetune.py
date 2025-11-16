import os
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

# ==== FIXED IMPORTS ====
from finetuning.text_encoder import CLIPTextEncoder
from networks.DiT.dit import STDiT
from data.covla_dataset import CoVLAOrbisMultiFrame
# ======================

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 2
LR = 1e-5
EPOCHS = 1
CONTEXT_FRAMES = 4
TARGET_FRAMES = 4
DIFF_STEPS = 1000

# ===================== Dataset =====================
train_ds = CoVLAOrbisMultiFrame(
    split="train",
    num_frames=CONTEXT_FRAMES + TARGET_FRAMES,
    streaming=True,
    captions_dir="data/covla_captions",
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

# ===================== Text Encoder =====================
text_encoder = CLIPTextEncoder(device=device).to(device)
for p in text_encoder.model.parameters():
    p.requires_grad = False

# ===================== Model =====================
model = STDiT(
    input_size=[16, 16],
    patch_size=2,
    in_channels=3,
    hidden_size=1152,
    depth=28,
    num_heads=16,
    mlp_ratio=4.0,
    max_num_frames=8,
    dropout=0.0,
).to(device)

ckpt_path = os.path.join("checkpoints", "orbis_288x512.ckpt")
ckpt = torch.load(ckpt_path, map_location="cpu")
missing, unexpected = model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(text_encoder.proj.parameters()),
    lr=LR,
    weight_decay=0.01
)

step = 0
model.train()

# ===================== Training Loop =====================
for epoch in range(EPOCHS):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        imgs = batch["images"].to(device)
        captions = batch["caption"]

        # frame rate broadcast
        frame_rate = torch.tensor(
            [batch["frame_rate"]] * imgs.size(0),
            device=device,
            dtype=torch.float32
        )

        B, F, _, _, _ = imgs.shape
        assert F == CONTEXT_FRAMES + TARGET_FRAMES

        context = imgs[:, :CONTEXT_FRAMES]
        target = imgs[:, CONTEXT_FRAMES:]

        # build 1 caption per video
        video_caps = []
        for cap_list in captions:
            non_empty = [c.strip() for c in cap_list if c.strip() != ""]
            video_caps.append(" ".join(non_empty) if non_empty else "no caption")

        text_emb = text_encoder(video_caps)

        t = torch.randint(0, DIFF_STEPS, (B,), device=device)
        noise = torch.randn_like(target)
        noisy_target = target + noise

        pred = model(
            target=noisy_target,
            context=context,
            t=t,
            frame_rate=frame_rate,
            text_emb=text_emb,
        )

        loss = torch.mean((pred - noise) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step += 1
        pbar.set_postfix({"loss": loss.item()})

# ===================== Save Model =====================
save_path = "finetuned_orbis_text.ckpt"
torch.save({
    "model": model.state_dict(),
    "text_proj": text_encoder.proj.state_dict(),
}, save_path)

print(f"\n🔥 FINISHED — saved: {save_path}")
