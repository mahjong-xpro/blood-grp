"""SuitAwareResNet encoder for Bloody Battle Mahjong.

Convolutions respect suit boundaries (Man/Pin/Sou each 9 tiles).
Uses shared weights across all 3 suits for parameter efficiency and
inductive bias (suits are structurally isomorphic).
"""

import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace

TILES_PER_SUIT = 9
NUM_TILES = 27
DEFAULT_OBS_CHANNELS = 384


class SuitAwareConv1d(nn.Module):
    """Conv1d that processes each suit independently with shared weights, parallelized."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        nn.init.orthogonal_(self.conv.weight, gain=1.414)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, 27)
        B, C, T = x.shape
        # Reshape to (B*3, C, 9) to process all suits in one kernel call
        x = x.view(B, C, 3, TILES_PER_SUIT).permute(0, 2, 1, 3).reshape(B * 3, C, TILES_PER_SUIT)
        x = self.conv(x)
        # Reshape back to (B, out_C, 27)
        out_C = x.shape[1]
        x = x.view(B, 3, out_C, TILES_PER_SUIT).permute(0, 2, 1, 3).reshape(B, out_C, T)
        return x


class SuitPositionalEncoding(nn.Module):
    """Learnable rank-position embedding, shared across suits.

    Adds a (channels, 9) embedding tiled across all 3 suits to give the
    model explicit awareness of tile rank (1-9). Suits are isomorphic so
    the same embedding is reused for Man, Pin, and Sou.
    """

    def __init__(self, channels: int, tiles_per_suit: int = TILES_PER_SUIT):
        super().__init__()
        # One embedding vector per rank position, shared across suits
        self.pos_embed = nn.Parameter(torch.zeros(channels, tiles_per_suit))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, 27)
        embed = self.pos_embed.repeat(1, 3)   # (C, 27) — tile across 3 suits
        return x + embed.unsqueeze(0)          # broadcast over batch


class BottleneckBlock(nn.Module):
    """Bottleneck ResBlock: 1x1 (down) -> 3x3 (SuitAware) -> 1x1 (up)."""

    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        mid_channels = channels // expansion
        ng = _num_groups(channels)
        mid_ng = _num_groups(mid_channels)

        self.block = nn.Sequential(
            nn.GroupNorm(ng, channels),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, mid_channels, 1, bias=False),  # 1x1 down
            nn.GroupNorm(mid_ng, mid_channels),
            nn.Mish(inplace=True),
            SuitAwareConv1d(mid_channels, mid_channels, kernel_size=3),
            nn.GroupNorm(mid_ng, mid_channels),
            nn.Mish(inplace=True),
            nn.Conv1d(mid_channels, channels, 1, bias=False),  # 1x1 up
        )
        self.attn = ChannelAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.block(x))


class ChannelAttention(nn.Module):
    """Squeeze-Excitation channel attention for cross-suit information exchange."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.Mish(inplace=True),
            nn.Linear(mid, channels),
        )
        self.max_fc = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.Mish(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.sigmoid(self.avg_fc(x) + self.max_fc(x)).unsqueeze(-1)
        return x * scale


class TileAttention(nn.Module):
    """Multi-head self-attention over 27 tile positions.

    Allows direct cross-suit tile interaction (e.g. Man-1 attending to Pin-1,
    or Man-5 attending to Sou-5) that the suit-isolated convolutions cannot model.
    Uses pre-norm + residual connection for training stability.
    """

    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, 27)
        x_t = x.permute(0, 2, 1)                              # (B, 27, C)
        x_norm = self.norm(x_t)                                # pre-norm
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return (x_t + attn_out).permute(0, 2, 1)              # (B, C, 27)


def _num_groups(channels: int, preferred: int = 16) -> int:
    """Find a valid number of groups for GroupNorm."""
    for g in [preferred, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """Pre-activation ResBlock with SuitAwareConv and ChannelAttention."""

    def __init__(self, channels: int):
        super().__init__()
        ng = _num_groups(channels)
        self.block = nn.Sequential(
            nn.GroupNorm(ng, channels),
            nn.Mish(inplace=True),
            SuitAwareConv1d(channels, channels, kernel_size=3),
            nn.GroupNorm(ng, channels),
            nn.Mish(inplace=True),
            SuitAwareConv1d(channels, channels, kernel_size=3),
        )
        self.attn = ChannelAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.block(x))


class SuitAwareResNetEncoder(Encoder):
    """SuitAwareResNet encoder for Sample Factory with Bottleneck blocks.

    Architecture:
        obs (B, C*27) -> reshape (B, C, 27)
        -> stem: SuitAwareConv(C -> conv_ch) + GroupNorm + Mish
        -> pos_enc: SuitPositionalEncoding (rank-aware embeddings)
        -> res_blocks: BottleneckBlock x num_blocks
        -> tile_attn: TileAttention (cross-suit tile interaction)
        -> flatten -> [B, conv_ch * 27]
    """

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.obs_channels = getattr(cfg, "blood_obs_channels", DEFAULT_OBS_CHANNELS)
        conv_ch = getattr(cfg, "blood_conv_channels", 256)
        num_blocks = getattr(cfg, "blood_num_res_blocks", 20)

        ng = _num_groups(conv_ch)

        self.stem = nn.Sequential(
            SuitAwareConv1d(self.obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)
        self.res_blocks = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(num_blocks)])
        self.tile_attn = TileAttention(conv_ch, num_heads=4)

        self._out_size = conv_ch * NUM_TILES

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        B = obs.shape[0]
        x = obs.view(B, self.obs_channels, NUM_TILES)
        x = self.stem(x)
        x = self.pos_enc(x)
        x = self.res_blocks(x)
        x = self.tile_attn(x)
        return x.reshape(B, -1)

    def get_out_size(self) -> int:
        return self._out_size
