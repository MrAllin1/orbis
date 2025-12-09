# finetuning/adaln_models.py
import os
import torch
from omegaconf import OmegaConf

from util import instantiate_from_config
from finetuning.text_encoder import CLIPTextEncoder
from models.second_stage.fm_model import ModelIF as FMWorldModel  # NEW


def create_text_encoder(device: torch.device):
    text_encoder = CLIPTextEncoder(device=device).to(device)
    text_encoder.eval()
    for p in text_encoder.model.parameters():
        p.requires_grad = False
    print("Text encoder initialized & frozen.")
    return text_encoder


def create_tokenizer(model_cfg, tokenizer_device: torch.device):
    """
    Kept for compatibility if you need the raw Stage1 AE somewhere else.
    Not used anymore in train_adaln_debug.py after switching to fm_model.Model.
    """
    print("\nLoading tokenizer from stage1 config...")

    tk_cfg = model_cfg["tokenizer_config"]
    tokenizer_folder = os.path.expandvars(tk_cfg["folder"])
    tokenizer_ckpt_path = os.path.join(tokenizer_folder, tk_cfg["ckpt_path"])
    tk_config_path = os.path.join(tokenizer_folder, "config.yaml")

    tk_config = OmegaConf.load(tk_config_path)

    # clean unsupported encoder kwargs
    try:
        enc_params = tk_config["model"]["params"]["encoder_config"]["params"]
        if "use_pretrained_weights" in enc_params:
            enc_params.pop("use_pretrained_weights")
    except Exception:
        pass

    tokenizer = instantiate_from_config(tk_config["model"]).to(tokenizer_device)

    tk = torch.load(tokenizer_ckpt_path, map_location="cpu")
    tokenizer.load_state_dict(tk["state_dict"], strict=True)
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    print("Tokenizer ready.")
    return tokenizer


def create_world_model(model_cfg, max_frames: int, device: torch.device, orbit_ckpt_path: str):
    """
    Adapter that instantiates the *full* FM world model from models/second_stage/fm_model.py

    This gives you:
      - model.vit        -> STDiT backbone (where we plug AdaLN + text)
      - model.ae         -> Stage1 tokenizer (VQ)
      - model.add_noise  -> correct alpha/sigma schedule
      - model.encode_frames / decode_frames
      - model.roll_out / sample for generation
    """
    print("\nLoading FM world model (Model from fm_model.py)...")

    # Wrap tokenizer_config & generator_config into config objects fm_model expects
    tk_cfg = OmegaConf.create(model_cfg["tokenizer_config"])
    gen_cfg = OmegaConf.create(model_cfg["generator_config"])

    world_model = FMWorldModel(
        tokenizer_config=tk_cfg,
        generator_config=gen_cfg,
        adjust_lr_to_batch_size=model_cfg.get("adjust_lr_to_batch_size", False),
        sigma_min=model_cfg.get("sigma_min", 1e-5),
        timescale=model_cfg.get("timescale", 1.0),
        enc_scale=model_cfg.get("enc_scale", 4.0),
        warmup_steps=model_cfg.get("warmup_steps", 5000),
        min_lr_multiplier=model_cfg.get("min_lr_multiplier", 0.1),
    )

    # Load Stage2 FM checkpoint
    ckpt = torch.load(orbit_ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = world_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded world model ckpt. Missing keys: {len(missing)}, unexpected: {len(unexpected)}")

    world_model = world_model.to(device)
    print("World model ready (STDiT backbone, tokenizer, noise schedule, sampling).")
    return world_model
