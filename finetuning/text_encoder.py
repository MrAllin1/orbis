import torch
import clip
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        self.device = device

        # Project CLIP → hidden_size (Orbis hidden size ≈ 768)
        self.proj = nn.Linear(512, 768)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, captions):
        """
        captions: list(str) of length B  (one string per video)
        output:   (B, 1152)
        """
        tokens = clip.tokenize(captions).to(self.device)   # (B, T)
        with torch.no_grad():
            clip_emb = self.model.encode_text(tokens)      # (B, 512)

        return self.proj(clip_emb)                         # (B, 768)
