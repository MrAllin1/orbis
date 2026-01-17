from numpy import short
import torch
import clip
import torch.nn as nn

class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = torch.device(device)

        # Load CLIP and freeze it
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Project CLIP → hidden_size (Orbis hidden size ≈ 768)
        self.proj = nn.Linear(512, 768)

        # AdaLN-Zero style: start from zero (no conditioning at step 0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, captions):
        """
        captions: list(str) of length B  (one string per video)
        output:   (B, 768), on self.device
        """
        # Tokenize on the same device as CLIP
        try:
            tokens = clip.tokenize(captions).to(self.device)
        except RuntimeError:
            # crude but safe fallback: shorten each caption and retry
            short = []
            for t in captions:
                t = t if isinstance(t, str) else ""
                short.append(" ".join(t.split()[:40]))
            tokens = clip.tokenize(short).to(self.device)

        with torch.no_grad():
            clip_emb = self.model.encode_text(tokens)      # (B, 512)

        # Match Linear weights dtype (usually fp32)
        clip_emb = clip_emb.to(self.proj.weight.dtype)

        return self.proj(clip_emb)                         # (B, 768)
