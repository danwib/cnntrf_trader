import torch
import torch.nn as nn

class ChannelMLPMixer(nn.Module):
    def __init__(self, channels: int, hidden: int = None, dropout: float = 0.1):
        super().__init__()
        hidden = hidden or max(32, channels // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        self.ln = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    def forward(self, x):  # (B, C, L)
        return self.ln(x + self.net(x))

class ChannelIVarAttention(nn.Module):
    def __init__(self, channels: int, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out_proj = nn.Linear(d_model, 1)
        self.channels = channels

    def forward(self, x):  # (B, C, L)
        B, C, L = x.shape
        s = x.mean(dim=-1, keepdim=True)       # (B, C, 1)
        s = self.in_proj(s)                    # (B, C, d_model)
        z = self.encoder(s)                    # (B, C, d_model)
        z = self.out_proj(z)                   # (B, C, 1)
        z = z.expand(B, C, L)
        return x + z

class ChannelMixer(nn.Module):
    def __init__(self, channels: int, kind="mlp", dropout=0.1):
        super().__init__()
        if kind == "mlp":
            self.mod = ChannelMLPMixer(channels, dropout=dropout)
        elif kind == "ivar":
            self.mod = ChannelIVarAttention(channels, dropout=dropout)
        else:
            self.mod = nn.Identity()

    def forward(self, x):
        return self.mod(x)
