import torch
import torch.nn as nn
import clip

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device

        self.proj = nn.Linear(512, 1152)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, captions):
        tokens = clip.tokenize(captions).to(self.device)
        with torch.no_grad():
            clip_emb = self.model.encode_text(tokens)

        return self.proj(clip_emb)
