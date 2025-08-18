import torch
import torch.nn as nn

class PatchTokenizer(nn.Module):
    """
    Time-patching tokenizer: turns (B, C, L) into tokens (B, T, D).
    Implemented as a Conv1d along the time dimension with stride=patch_stride and kernel_size=patch_size.
    """
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 8, patch_stride: int = 4):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=0,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        returns: tokens (B, T, D)
        """
        z = self.proj(x)      # (B, D, T)
        z = z.transpose(1, 2) # (B, T, D)
        return z
