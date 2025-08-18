import torch
import torch.nn as nn

def causal_mask(T: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

class TimeTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, num_layers=3, dropout=0.1, causal=True):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.causal = causal

    def forward(self, x):
        if self.causal:
            mask = causal_mask(x.size(1), x.device)
            return self.encoder(x, mask=mask)
        return self.encoder(x)
