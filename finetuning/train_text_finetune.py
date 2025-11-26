import os
import torch
import torch.nn as nn
from tqdm import tqdm

# ==== PROJECT IMPORTS ====
from finetuning.text_encoder import CLIPTextEncoder
from data.covla_dataset import CoVLAOrbisMultiFrame
from models.first_stage.vqgan import VQModelIF as Tokenizer
from networks.DiT.dit import SwinSTDiTNoExtraMLP as STDiT
# ===============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ====== PATHS ======
TOKENIZER_CKPT = "/home/fr/fr_fr/fr_aa533/work/orbis/logs_tk/tokenizer_288x512/checkpoints/tokenizer_288x512.ckpt"
ORBIT_CKPT     = "/home/fr/fr_fr/fr_aa533/work/orbis/logs_wm/orbis_288x512/checkpoints/last.ckpt"
CAPTIONS_DIR   = "/home/fr/fr_fr/fr_aa533/work/orbis/data/covla_captions"
SAVE_PATH      = "/home/fr/fr_fr/fr_aa533/work/orbis/finetuning/finetuned_orbis_AdaLN.ckpt"

# ====== HYPERPARAMS ======
BATCH_SIZE      = 1
LR              = 1e-4
EPOCHS          = 1
STEPS_PER_EPOCH = 200
CONTEXT_FRAMES  = 3
TARGET_FRAMES   = 3
DIFF_STEPS      = 1000
MAX_FRAMES      = CONTEXT_FRAMES + TARGET_FRAMES
beta_start = 0.0001
beta_end   = 0.02
betas = torch.linspace(beta_start, beta_end, DIFF_STEPS, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
# ===============================================================

# ===================== Dataset =====================
train_ds = CoVLAOrbisMultiFrame(
    split="train",
    num_frames=MAX_FRAMES,
    streaming=True,
    captions_dir=CAPTIONS_DIR,
)

def infinite_stream(ds):
    it = iter(ds)
    while True:
        try:
            yield next(it)
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
#                 TOKENIZER: EXACT YAML CONFIG
# ===============================================================
print("\n🔍 Loading Tokenizer with correct YAML config...")

class ConfigDict(dict):
    def __init__(self, target, params):
        super().__init__(target=target, params=params)
        self.target = target
        self.params = params


encoder_cfg = ConfigDict(
    "networks.tokenizer.pretrained_models.Encoder",
    {
        "resolution": [288, 512],
        "patch_size": 16,
        "z_channels": 768,
        "pretrained_encoder": "MAE",
        "normalize_embedding": True,
    }
)

decoder_cfg = ConfigDict(
    "networks.tokenizer.ae.Decoder",
    {
        "double_z": False,
        "z_channels": 768,
        "resolution": 256,   # From tokenizer YAML — not list
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "normalize_embedding": False,
    }
)

quant_cfg = ConfigDict(
    "modules.quantize.VectorQuantizer",
    {
        "n_e": 16384,
        "e_dim": 16,
        "beta": 0.25,
        "normalize_embedding": True,
    }
)

loss_cfg = ConfigDict(
    "modules.vqloss.VQLPIPSWithDiscriminator",
    {
        "disc_conditional": False,
        "disc_in_channels": 3,
        "disc_start": 10000,
        "disc_weight": 0.1,
        "adaptive_disc_weight": True,
        "codebook_weight": 1.0,
        "distill_loss_weight": 2.0,
        "perceptual_weight": 1.0,
        "l1_loss_weight": 1.0,
        "l2_loss_weight": 0.0,
        "warmup_steps": 5000,
        "beta_1": 0.5,
        "beta_2": 0.9,
    }
)

entropy_cfg = ConfigDict(
    "modules.lr_scheduler.VQEntropyLossScheduler",
    {
        "decay_steps": 10000,
        "weight_max": 0.01,
        "weight_min": 0.001,
    }
)

tokenizer = Tokenizer(
    encoder_config=encoder_cfg,
    decoder_config=decoder_cfg,
    quantizer_config=quant_cfg,
    loss_config=loss_cfg,
    entropy_loss_weight_scheduler_config=entropy_cfg,
).to(device)

print(f"Loading tokenizer weights from: {TOKENIZER_CKPT}")
tk = torch.load(TOKENIZER_CKPT, map_location="cpu")
tokenizer.load_state_dict(tk["state_dict"], strict=False)
tokenizer.eval()
for p in tokenizer.parameters():
    p.requires_grad = False
print("Tokenizer ready ✓")


# ===================== STDiT =====================
print("\n🎯 Loading STDiT (AdaLN training only)...")
model = STDiT(
    input_size=[18, 32],
    patch_size=1,
    in_channels=32,
    hidden_size=768,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    max_num_frames=MAX_FRAMES,
    dropout=0.1,
).to(device)

stdit_ckpt = torch.load(ORBIT_CKPT, map_location="cpu")
model.load_state_dict(stdit_ckpt.get("state_dict", stdit_ckpt), strict=False)
print("STDiT pretrained weights loaded.")

for name, p in model.named_parameters():
    p.requires_grad = ("adaLN" in name)

for p in text_encoder.proj.parameters():
    p.requires_grad = True

train_params = [p for p in model.parameters() if p.requires_grad] + \
               list(text_encoder.proj.parameters())

optimizer = torch.optim.AdamW(train_params, lr=LR, weight_decay=0.01)
print(f"Trainable parameters: {sum(p.numel() for p in train_params):,}")


# ================ TOKENIZER LATENT ENCODING =================
def encode_latents(imgs: torch.Tensor) -> torch.Tensor:
    B, F, C, H, W = imgs.shape
    imgs = imgs.reshape(B * F, C, H, W)

    with torch.no_grad():
        z = tokenizer.encode_first_stage(imgs)
        latents, _, _ = tokenizer.quantize(z)

    return latents.view(B, F, 32, 18, 32)


# ===================== TRAIN =====================
print("\n🚀 Starting AdaLN Fine-Tuning...\n")
model.train()

for epoch in range(EPOCHS):
    pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}")

    for _ in pbar:
        s = next(data_stream)

        imgs = s["images"].unsqueeze(0).to(device)
        latents = encode_latents(imgs)

        context = latents[:, :CONTEXT_FRAMES]
        target  = latents[:, CONTEXT_FRAMES:]

        # ---- CAPTION FIX ----
        text = s["caption"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "no caption"
        text_emb = text_encoder([text]).to(device)

        # ---- TIMESTEP ----
        t = torch.randint(0, DIFF_STEPS, (1,), device=device)

        # ---- CORRECT DDPM NOISE ----
        alpha_bar = alphas_cumprod[t].view(1, 1, 1, 1, 1)
        noise = torch.randn_like(target)
        noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1 - alpha_bar) * noise

        frame_rate = torch.tensor([s["frame_rate"]], device=device)

        pred = model(noisy, context, t, frame_rate, text_emb=text_emb)
        loss = torch.mean((pred - noise)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=float(loss))


# ================= SAVE =================
torch.save({
    "state_dict": model.state_dict(),
    "text_proj": text_encoder.proj.state_dict(),
}, SAVE_PATH)

print(f"\n🔥 DONE! AdaLN Fine-Tune Saved → {SAVE_PATH}")
