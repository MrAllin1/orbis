# finetuning/adaln_config.py
import os
import yaml
from dataclasses import dataclass
from omegaconf import OmegaConf
import torch

# Your repo
BASE_WORK_UNIS = "/work/dlclarge2/alidemaa-text-control-orbis/orbis"
BASE_WORK_KIT = "/home/fr/fr_fr/fr_aa533/work/orbis"  # if you ever want to switch

# Path to your fine-tuning FM config (ModelIF + CoVLA)
FM_CONFIG_PATH = f"{BASE_WORK_UNIS}/finetuning/fm_finetune_covla_modelif.yaml"

# === LOCAL CHECKPOINTS (your copies) ===

# Stage2 FM world model (ModelIF)
ORBIT_CKPT = (
    f"{BASE_WORK_UNIS}/logs_wm/orbis_288x512/checkpoints/last.ckpt"
)

# Stage1 tokenizer (VQ, IF-style latents)
TOKENIZER_ROOT = f"{BASE_WORK_UNIS}/logs_tk/tokenizer_192x336"
TOKENIZER_CKPT_REL = "checkpoints/epoch-26_rfid_8_9.ckpt"

# === OUTPUT PATHS FOR ADALN FINE-TUNE ===
SAVE_PATH       = "/work/dlclarge2/alidemaa-text-control-orbis/orbis/finetuning/balanced_train.ckpt"
CHECKPOINT_PATH = f"{BASE_WORK_UNIS}/finetuning/checkpoints/adaln_text_conditioning_balanced_train.ckpt"


@dataclass
class TrainHyperparams:
    batch_size: int = 1
    lr: float = 1e-4
    epochs: int =10
    steps_per_epoch: int = 3708
    val_steps_per_epoch: int = 412   # NEW
    context_frames: int = 4
    target_frames: int = 2

    beta_start: float = 0.0001
    beta_end: float = 0.02
    diffusion_steps: int = 1000

def load_fm_config():
    """Load YAML config and return (fm_cfg, model_cfg, train_params, val_params)."""
    print(f"Loading FM config from: {FM_CONFIG_PATH}")
    with open(FM_CONFIG_PATH, "r") as f:
        fm_cfg = yaml.safe_load(f)

    # Model params (used by create_world_model → ModelIF)
    model_cfg = fm_cfg["model"]["params"]
    data_cfg = fm_cfg["data"]["params"]

    # --- OVERRIDE TOKENIZER PATHS TO YOUR LOCAL COPY ---
    if "tokenizer_config" in model_cfg:
        tk_cfg = model_cfg["tokenizer_config"]
        tk_cfg["folder"] = TOKENIZER_ROOT
        tk_cfg["ckpt_path"] = TOKENIZER_CKPT_REL
        print("[FM CONFIG] Overriding tokenizer_config with local paths:")
        print(f"  folder    = {tk_cfg['folder']}")
        print(f"  ckpt_path = {tk_cfg['ckpt_path']}")

    # ---------------- TRAIN PARAMS ----------------
    train_cfg = data_cfg["train"]
    if isinstance(train_cfg, list):
        train_entry = train_cfg[0]
    else:
        train_entry = train_cfg
    train_target = train_entry["target"]
    train_params = dict(train_entry["params"])  # copy to allow setdefault

    print(f"Train dataset target from YAML: {train_target}")

    captions_default = f"{BASE_WORK_UNIS}/data/covla_captions_balanced"
    videos_default   = f"{BASE_WORK_UNIS}/data/covla_videos_balanced"

    train_params.setdefault("captions_dir", captions_default)
    train_params.setdefault("videos_dir",   videos_default)

    # ---------------- VAL PARAMS ----------------
    val_cfg = data_cfg.get("validation", None)
    val_params = None
    if val_cfg is not None:
        val_entry = val_cfg
        val_target = val_entry["target"]
        val_params = dict(val_entry["params"])
        print(f"Validation dataset target from YAML: {val_target}")

        # Default captions_dir if missing
        val_params.setdefault("captions_dir", captions_default)
        # videos_dir is already in your YAML for validation

    print("YAML data params (train / CoVLA):")
    print(f"  size            = {train_params['size']}")
    print(f"  num_frames      = {train_params['num_frames']}")
    print(f"  stored_rate     = {train_params.get('stored_data_frame_rate', 20)}")
    print(f"  target_rate     = {train_params.get('target_frame_rate', 5)}")
    print(f"  captions_dir    = {train_params['captions_dir']}")
    print(f"  videos_dir      = {train_params['videos_dir']}")

    if val_params is not None:
        print("YAML data params (val / CoVLA):")
        print(f"  size            = {val_params['size']}")
        print(f"  num_frames      = {val_params['num_frames']}")
        print(f"  stored_rate     = {val_params.get('stored_data_frame_rate', 20)}")
        print(f"  target_rate     = {val_params.get('target_frame_rate', 5)}")
        print(f"  captions_dir    = {val_params['captions_dir']}")
        print(f"  videos_dir      = {val_params['videos_dir']}")

    print("\n[FM CONFIG] Using Stage2 FM checkpoint:")
    print(f"  ORBIT_CKPT = {ORBIT_CKPT}\n")

    return fm_cfg, model_cfg, train_params, val_params


def make_diffusion_schedule(beta_start, beta_end, num_steps, device):
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod
