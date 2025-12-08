# finetuning/adaln_utils.py
import os
import torch


def infinite_stream(ds):
    it = iter(ds)
    while True:
        try:
            yield next(it)
        except StopIteration:
            it = iter(ds)
        except Exception:
            it = iter(ds)


def encode_latents(imgs: torch.Tensor, tokenizer, tokenizer_device, device):
    """
    imgs: (B, F, C, H, W) on CPU
    returns: (B, F, C_lat, H_lat, W_lat) on main device (GPU)
    """
    B, F, C, H, W = imgs.shape
    imgs = imgs.reshape(B * F, C, H, W).to(tokenizer_device)

    with torch.no_grad():
        encoded = tokenizer.encode(imgs)
        quantized = encoded["quantized"]

        if isinstance(quantized, tuple):
            q_rec, q_sem = quantized
            lat = torch.cat([q_rec, q_sem], dim=1)
        else:
            lat = quantized

    _, C_lat, H_lat, W_lat = lat.shape
    latents = lat.view(B, F, C_lat, H_lat, W_lat)
    latents = latents.to(device)
    return latents


def save_checkpoint(path, epoch, step, model, text_encoder, optimizer):
    tmp_path = path + ".tmp"
    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "text_proj_state_dict": text_encoder.proj.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)
    print(f"[CKPT] Saved checkpoint at epoch={epoch}, step={step}", flush=True)


def load_checkpoint_if_exists(path, model, text_encoder, optimizer, device):
    start_epoch = 0
    if not os.path.exists(path):
        print("[CKPT] No existing checkpoint found → starting from scratch")
        return start_epoch

    print(f"[CKPT] Found existing checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        text_encoder.proj.load_state_dict(ckpt["text_proj_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        last_epoch = ckpt.get("epoch", 0)
        start_epoch = last_epoch + 1
        print(f"[CKPT] Resuming from epoch={start_epoch}")
    except Exception as e:
        print(f"[CKPT] Failed to load checkpoint ({e}) → starting from scratch")
        start_epoch = 0

    return start_epoch
