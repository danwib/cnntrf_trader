from dataclasses import dataclass, asdict
import torch
import torch.nn as nn

from .modules.cnn_stem import CNNStem
from .modules.patch_tokenizer import PatchTokenizer
from .modules.positional import SinePositionalEncoding
from .modules.time_transformer import TimeTransformer
from .modules.channel_mixer import ChannelMixer

@dataclass
class CNNPatchTSTConfig:
    in_channels: int
    num_symbols: int
    seq_len: int
    cnn_hidden: int = 128
    cnn_blocks: int = 3
    cnn_kernels: tuple = (5,5,3)
    cnn_dilations: tuple = (1,2,4)
    cnn_dropout: float = 0.1
    channel_mixer: str = "mlp"   # "mlp" | "ivar" | "none"
    patch_size: int = 8
    patch_stride: int = 4
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 3
    trf_dropout: float = 0.1
    emb_dim: int = 12
    pool: str = "last"           # "last" | "mean"

class CNNPatchTST(nn.Module):
    def __init__(self, cfg: CNNPatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        C_in = cfg.in_channels
        if cfg.emb_dim > 0:
            self.sym_emb = nn.Embedding(cfg.num_symbols, cfg.emb_dim)
            C_in += cfg.emb_dim
        else:
            self.sym_emb = None

        self.cnn = CNNStem(C_in, cfg.cnn_hidden, cfg.cnn_blocks,
                           list(cfg.cnn_kernels), list(cfg.cnn_dilations), cfg.cnn_dropout)
        C_after = cfg.cnn_hidden
        self.mixer = ChannelMixer(C_after, kind=cfg.channel_mixer)
        self.tokenizer = PatchTokenizer(C_after, cfg.d_model, cfg.patch_size, cfg.patch_stride)
        self.posenc = SinePositionalEncoding(cfg.d_model)
        self.trf = TimeTransformer(cfg.d_model, cfg.n_heads, cfg.num_layers, cfg.trf_dropout)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, x, sym_id=None):
        B, C, L = x.shape
        if self.sym_emb is not None and sym_id is not None:
            e = self.sym_emb(sym_id).unsqueeze(-1).expand(B, self.cfg.emb_dim, L)
            x = torch.cat([x, e], dim=1)

        z = self.cnn(x)
        z = self.mixer(z)
        tok = self.tokenizer(z)
        tok = self.posenc(tok)
        tok = self.trf(tok)
        tok = self.norm(tok)
        pooled = tok[:, -1] if self.cfg.pool == "last" else tok.mean(dim=1)
        return self.head(pooled)

    def config_dict(self):
        return asdict(self.cfg)
