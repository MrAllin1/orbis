# finetuning/adaln_config.py
import os
import yaml
from dataclasses import dataclass
from omegaconf import OmegaConf

# Your repo
BASE_WORK_UNIS = "/work/dlclarge2/alidemaa-text-control-orbis/orbis"
BASE_WORK_KIT = "/home/fr/fr_fr/fr_aa533/work/orbis"  # if you ever want to switch

# Shared dir with tokenizer + fm weights
TK_WORK_DIR = os.environ.get(
    "TK_WORK_DIR",
    "/work/dlclarge2/dienertj-orbisshare"
)

FM_CONFIG_PATH = f"{BASE_WORK_UNIS}/finetuning/stage2_baseline_covla_bev.yaml"

# Use the new Stage2 FM checkpoint from logs_fm
ORBIT_CKPT = (
    f"{TK_WORK_DIR}/logs_fm/"
    "2025-09-11T18-50-33_stage2_vq_if_192x336_DLC12406894/"
    "checkpoints/last.ckpt"
)

SAVE_PATH       = f"{BASE_WORK_UNIS}/finetuning/finetuned_orbis_AdaLN.ckpt"
CHECKPOINT_PATH = f"{BASE_WORK_UNIS}/finetuning/checkpoints/adaln_debug.ckpt"


@dataclass
class TrainHyperparams:
    batch_size: int = 1
    lr: float = 1e-4
    epochs: int = 30
    steps_per_epoch: int = 200

    context_frames: int = 3
    target_frames: int = 3

    beta_start: float = 0.0001
    beta_end: float = 0.02
    diffusion_steps: int = 1000


def load_fm_config():
    """Load YAML config and return (fm_cfg, model_cfg, train_params, train_ds_meta)."""
    print(f"Loading FM config from: {FM_CONFIG_PATH}")
    with open(FM_CONFIG_PATH, "r") as f:
        fm_cfg = yaml.safe_load(f)

    model_cfg = fm_cfg["model"]["params"]
    data_cfg = fm_cfg["data"]["params"]

    train_entry = data_cfg["train"][0]
    train_target = train_entry["target"]
    train_params = dict(train_entry["params"])  # copy to allow setdefault

    print(f"Train dataset target from YAML: {train_target}")

    # defaults for paths if not present in YAML
    captions_default = f"{BASE_WORK_UNIS}/data/covla_captions"
    videos_default   = f"{BASE_WORK_UNIS}/data/covla_100_videos"

    train_params.setdefault("captions_dir", captions_default)
    train_params.setdefault("videos_dir",   videos_default)

    # log some debug info
    print("YAML data params (train / CoVLA):")
    print(f"  size            = {train_params['size']}")
    print(f"  num_frames      = {train_params['num_frames']}")
    print(f"  stored_rate     = {train_params.get('stored_data_frame_rate', 20)}")
    print(f"  target_rate     = {train_params.get('target_frame_rate', 5)}")
    print(f"  scale_min/max   = {train_params.get('scale_min', 0.75)}, {train_params.get('scale_max', 1.0)}")
    print(f"  captions_dir    = {train_params['captions_dir']}")
    print(f"  videos_dir      = {train_params['videos_dir']}")

    return fm_cfg, model_cfg, train_params


def make_diffusion_schedule(beta_start, beta_end, num_steps, device):
    import torch

    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod
