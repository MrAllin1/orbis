import os
import time

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
BASE_WORK_UNIS = "/work/dlclarge2/alidemaa-text-control-orbis/orbis"
BASE_WORK_KIS = "/home/fr/fr_fr/fr_aa533/work/orbis"
TOKENIZER_CKPT = f"{BASE_WORK_UNIS}/logs_tk/tokenizer_288x512/checkpoints/tokenizer_288x512.ckpt"
ORBIT_CKPT     = f"{BASE_WORK_UNIS}/logs_wm/orbis_288x512/checkpoints/last.ckpt"
CAPTIONS_DIR   = f"{BASE_WORK_UNIS}/data/covla_captions"
VIDEOS_DIR     = f"{BASE_WORK_UNIS}/data/covla_100_videos"
SAVE_PATH      = f"{BASE_WORK_UNIS}/finetuning/finetuned_orbis_AdaLN.ckpt"
CHECKPOINT_PATH = f"{BASE_WORK_UNIS}/finetuning/checkpoints/adaln_resume.ckpt"

os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

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
    num_frames=MAX_FRAMES,
    captions_dir=CAPTIONS_DIR,
    videos_dir=VIDEOS_DIR,   # local mp4s
    num_samples=None,        # or cap it (e.g. 100)
    debug=False,             # set True for dataset spam
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
        "resolution": 256,
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
print("Tokenizer type:", type(tokenizer))

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


# =========================================================
#              CHECKPOINT LOAD / SAVE
# =========================================================
def save_checkpoint(epoch, step):
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "text_proj_state_dict": text_encoder.proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, CHECKPOINT_PATH)
    print(f"[CKPT] Saved checkpoint at epoch={epoch}, step={step} → {CHECKPOINT_PATH}")


start_epoch = 0
start_step = 0

if os.path.exists(CHECKPOINT_PATH):
    print(f"[CKPT] Found existing checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    text_encoder.proj.load_state_dict(ckpt["text_proj_state_dict"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    start_step = ckpt.get("step", 0) + 1  # continue from next step

    print(f"[CKPT] Resuming from epoch={start_epoch}, step={start_step}")
else:
    print("[CKPT] No existing checkpoint found → starting from scratch")


# ================ TOKENIZER LATENT ENCODING =================
def encode_latents(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (B, F, C, H, W)
    returns: (B, F, 32, 18, 32)
    """
    print("[ENCODE_LATENTS] Starting latent encoding...")
    B, F, C, H, W = imgs.shape
    print(f"[ENCODE_LATENTS] Input shape: B={B}, F={F}, C={C}, H={H}, W={W}")

    imgs = imgs.reshape(B * F, C, H, W)
    print(f"[ENCODE_LATENTS] Reshaped to: {(B * F, C, H, W)}")

    with torch.no_grad():
        print("[ENCODE_LATENTS] Calling tokenizer.encode(...)")
        encoded = tokenizer.encode(imgs)  # dict from VQModelIF.encode
        quantized = encoded["quantized"]

        if isinstance(quantized, tuple):
            q_rec, q_sem = quantized
            print("[ENCODE_LATENTS] Got tuple quantized: ",
                  f"q_rec shape={tuple(q_rec.shape)}, q_sem shape={tuple(q_sem.shape)}")
            lat = torch.cat([q_rec, q_sem], dim=1)  # (B*F, 32, 18, 32)
        else:
            print("[ENCODE_LATENTS] quantized is not a tuple, using as-is")
            lat = quantized

        print("[ENCODE_LATENTS] Latents after concat shape:", tuple(lat.shape))

    latents = lat.view(B, F, 32, 18, 32)
    print("[ENCODE_LATENTS] Final latent shape:", tuple(latents.shape))
    return latents


# ===================== TRAIN =====================
print("\n🚀 Starting AdaLN Fine-Tuning...\n")
model.train()

for epoch in range(start_epoch, EPOCHS):
    pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"Epoch {epoch}")

    for step in pbar:
        # skip previously finished steps when resuming
        if epoch == start_epoch and step < start_step:
            continue

        step_start = time.time()
        print(f"\n====================== E{epoch} STEP {step} ======================")

        # --------------------------------------------------------
        # 1) LOAD DATA
        # --------------------------------------------------------
        t0 = time.time()
        print("[STEP] Fetching sample from dataset...")
        s = next(data_stream)
        print(f"[OK] Sample loaded in {time.time() - t0:.3f} sec")
        print(f"       • video_id: {s.get('video_id', 'N/A')}")
        print(f"       • caption: {s['caption'][:60]}...")

        # --------------------------------------------------------
        # 2) MOVE IMAGES TO DEVICE
        # --------------------------------------------------------
        imgs = s["images"].unsqueeze(0)
        print("[STEP] Moving images to device...")
        t0 = time.time()
        imgs = imgs.to(device)
        print(f"[OK] Moved to {device} in {time.time() - t0:.3f} sec")

        # --------------------------------------------------------
        # 3) TOKENIZER → LATENTS
        # --------------------------------------------------------
        print("[STEP] Tokenizer: encode + concat codebooks...")
        t0 = time.time()
        latents = encode_latents(imgs)
        print(f"[OK] Tokenizer forward done in {time.time() - t0:.3f} sec")
        print(f"       • latents shape: {tuple(latents.shape)}")

        context = latents[:, :CONTEXT_FRAMES]
        target  = latents[:, CONTEXT_FRAMES:]

        # --------------------------------------------------------
        # 4) TEXT ENCODER
        # --------------------------------------------------------
        print("[STEP] Text encoder forward...")
        t0 = time.time()
        text = s["caption"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "no caption"
        text_emb = text_encoder([text]).to(device)
        print(f"[OK] CLIP done in {time.time() - t0:.3f} sec")

        # --------------------------------------------------------
        # 5) SAMPLE TIMESTEP + DDPM NOISE
        # --------------------------------------------------------
        print("[STEP] DDPM noise prep...")
        t0 = time.time()
        t = torch.randint(0, DIFF_STEPS, (1,), device=device)
        alpha_bar = alphas_cumprod[t].view(1, 1, 1, 1, 1)
        noise = torch.randn_like(target)
        noisy = torch.sqrt(alpha_bar) * target + torch.sqrt(1 - alpha_bar) * noise
        print(f"[OK] Noise prepared in {time.time() - t0:.3f} sec")

        # ======== FRAME RATE FROM DATASET (LOCAL) =========
        frame_rate = torch.tensor([train_ds.target_rate], device=device)

        # --------------------------------------------------------
        # 6) STDiT FORWARD
        # --------------------------------------------------------
        print("[STEP] STDiT forward...")
        t0 = time.time()
        pred = model(noisy, context, t, frame_rate, text_emb=text_emb)
        print(f"[OK] STDiT forward done in {time.time() - t0:.3f} sec")

        # --------------------------------------------------------
        # 7) LOSS + BACKWARD + OPTIMIZER
        # --------------------------------------------------------
        print("[STEP] Backward + optimizer...")
        t0 = time.time()
        loss = torch.mean((pred - noise)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[OK] Optimizer step in {time.time() - t0:.3f} sec")

        # --------------------------------------------------------
        # 8) SAVE CHECKPOINT (to survive 30-min jobs)
        # --------------------------------------------------------
        save_checkpoint(epoch, step)

        # --------------------------------------------------------
        # 9) TOTAL STEP TIME
        # --------------------------------------------------------
        total = time.time() - step_start
        print(f"================ E{epoch} STEP {step} DONE: {total:.3f} sec ================\n")

        pbar.set_postfix(loss=float(loss))


# ================= SAVE FINAL =================
torch.save({
    "state_dict": model.state_dict(),
    "text_proj": text_encoder.proj.state_dict(),
}, SAVE_PATH)

print(f"\n🔥 DONE! AdaLN Fine-Tune Saved → {SAVE_PATH}")
