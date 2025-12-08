# finetuning/adaln_models.py
import os
import torch
from omegaconf import OmegaConf

from util import instantiate_from_config
from finetuning.text_encoder import CLIPTextEncoder
from networks.DiT.dit import STDiT


def create_text_encoder(device: torch.device):
    text_encoder = CLIPTextEncoder(device=device).to(device)
    text_encoder.eval()
    for p in text_encoder.model.parameters():
        p.requires_grad = False
    print("Text encoder initialized & frozen.")
    return text_encoder


def create_tokenizer(model_cfg, tokenizer_device: torch.device):
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
    print("\nLoading STDiT (world model)...")

    gen_cfg = model_cfg["generator_config"]["params"]
    GEN_INPUT_SIZE  = gen_cfg["input_size"]
    GEN_IN_CHANNELS = gen_cfg["in_channels"]
    GEN_HIDDEN_SIZE = gen_cfg.get("hidden_size", 768)
    GEN_DEPTH       = gen_cfg.get("depth", 12)
    GEN_NUM_HEADS   = gen_cfg.get("num_heads", 12)
    GEN_MLP_RATIO   = gen_cfg.get("mlp_ratio", 4)
    GEN_DROPOUT     = gen_cfg.get("dropout", 0.0)
    GEN_LEARN_SIGMA = gen_cfg.get("learn_sigma", False)

    model = STDiT(
        input_size=GEN_INPUT_SIZE,
        patch_size=1,
        in_channels=GEN_IN_CHANNELS,
        hidden_size=GEN_HIDDEN_SIZE,
        depth=GEN_DEPTH,
        num_heads=GEN_NUM_HEADS,
        mlp_ratio=GEN_MLP_RATIO,
        max_num_frames=max_frames,
        dropout=GEN_DROPOUT,
        learn_sigma=GEN_LEARN_SIGMA,
    ).to(device)

    print("DEBUG: final_layer.linear mean abs BEFORE ckpt load:",
          model.final_layer.linear.weight.abs().mean().item())

    stdit_ckpt = torch.load(orbit_ckpt_path, map_location="cpu")
    model.load_state_dict(stdit_ckpt.get("state_dict", stdit_ckpt), strict=False)

    print("DEBUG: final_layer.linear mean abs AFTER ckpt load:",
          model.final_layer.linear.weight.abs().mean().item())

    print("STDiT pretrained weights loaded (strict=False).")
    return model
