
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinePositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (no parameters)."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to x.
        Args:
            x: shape (B, L, d_model)
        """
        L = x.size(1)
        return x + self.pe[:, :L]


class CausalConv1dSame(nn.Module):
    """1D convolution that uses left padding to preserve length and maintain causality (no peeking ahead)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left padding only: pad = (dilation*(k-1), 0)
        pad = self.dilation * (self.kernel_size - 1)
        x = F.pad(x, (pad, 0))
        return self.conv(x)


@dataclass
class CNNTransformerConfig:
    in_channels: int                  # feature channels (without embedding)
    num_symbols: int                  # for embedding
    emb_dim: int = 8                  # symbol embedding size
    conv_channels: Tuple[int, ...] = (64, 64)  # conv widths for the CNN stem
    kernel_sizes: Tuple[int, ...] = (5, 3)     # same length as conv_channels
    causal_conv: bool = False                 # if True, use causal padding
    use_bn: bool = True                       # BatchNorm after conv
    dropout: float = 0.1

    d_model: int = 128                        # transformer model size
    n_heads: int = 4
    num_layers: int = 3
    ff_ratio: float = 4.0
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1

    pooling: str = "last"                     # "last" or "mean"


class CNNTransformer(nn.Module):
    """
    CNN + Transformer for time-series regression on sequences (B, C, L).
    The symbol id is embedded and concatenated as extra channels across time.
    """
    def __init__(self, cfg: CNNTransformerConfig):
        super().__init__()
        self.cfg = cfg

        total_in = cfg.in_channels + cfg.emb_dim
        convs: List[nn.Module] = []
        in_ch = total_in
        for i, (ch, k) in enumerate(zip(cfg.conv_channels, cfg.kernel_sizes)):
            if cfg.causal_conv:
                conv = CausalConv1dSame(in_ch, ch, kernel_size=int(k))
            else:
                pad = int(k) // 2
                conv = nn.Conv1d(in_ch, ch, kernel_size=int(k), padding=pad)
            layers = [conv, nn.GELU()]
            if cfg.use_bn:
                layers.append(nn.BatchNorm1d(ch))
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            convs.append(nn.Sequential(*layers))
            in_ch = ch
        self.cnn = nn.Sequential(*convs) if convs else nn.Identity()

        # project to transformer dimension
        self.to_model = nn.Conv1d(in_ch, cfg.d_model, kernel_size=1)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=int(cfg.d_model * cfg.ff_ratio),
            dropout=cfg.ffn_dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.posenc = SinePositionalEncoding(cfg.d_model)

        # heads
        self.head = nn.Linear(cfg.d_model, 1)

        # embedding
        self.sym_embedding = nn.Embedding(cfg.num_symbols, cfg.emb_dim)

    def forward(self, x: torch.Tensor, sym_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) float
            sym_id: (B,) long/int
        Returns:
            y_hat: (B,) float
        """
        B, C, L = x.shape
        # symbol embedding broadcast as channels
        emb = self.sym_embedding(sym_id.long())  # (B, E)
        emb = emb.unsqueeze(-1).expand(B, emb.size(-1), L)  # (B, E, L)

        h = torch.cat([x, emb], dim=1)  # (B, C+E, L)
        h = self.cnn(h)                 # (B, C', L)
        h = self.to_model(h)            # (B, d_model, L)
        h = h.transpose(1, 2)           # (B, L, d_model)

        h = self.posenc(h)              # add positional encoding
        h = self.transformer(h)         # (B, L, d_model)

        if self.cfg.pooling == "mean":
            pooled = h.mean(dim=1)      # (B, d_model)
        else:
            pooled = h[:, -1, :]        # last timestep
        y_hat = self.head(pooled).squeeze(-1)
        return y_hat
