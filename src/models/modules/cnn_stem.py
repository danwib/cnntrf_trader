import torch
import torch.nn as nn
import torch.nn.functional as F

def _causal_pad_1d(x, kernel_size, dilation):
    pad_left = (kernel_size - 1) * dilation
    return F.pad(x, (pad_left, 0))

class DepthwiseSeparableTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dw = nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, bias=False)
        self.pw = nn.Conv1d(channels, channels * 2, kernel_size=1)  # *2 for GLU
        self.gn = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = _causal_pad_1d(x, self.kernel_size, self.dilation)
        x = self.dw(x)
        x = self.gn(x)
        x = F.glu(self.pw(x), dim=1)  # gated linear unit
        x = self.dropout(x)
        return x + residual

class CNNStem(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_blocks: int = 3,
                 kernel_sizes=None, dilations=None, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        if kernel_sizes is None: kernel_sizes = [5] * num_blocks
        if dilations is None: dilations = [1, 2, 4][:num_blocks]
        blocks = [DepthwiseSeparableTCNBlock(hidden_channels, kernel_sizes[i], dilations[i], dropout)
                  for i in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)
        self.out_norm = nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels)

    def forward(self, x):
        x = self.in_proj(x)
        x = self.blocks(x)
        x = self.out_norm(x)
        return x
