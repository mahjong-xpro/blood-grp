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

from blood.consts import (
    TILES_PER_SUIT,
    NUM_TILE_TYPES as NUM_TILES,
    NUM_STUDENT_CHANNELS as DEFAULT_OBS_CHANNELS,
)


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

    Includes a learnable position embedding over the 27 tile positions so that
    the attention is not permutation-invariant (self-attention alone has no
    notion of tile order without explicit positional information).
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
        # Learnable position embedding: one vector per tile position (27 total)
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_TILES, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, 27)
        x_t = x.permute(0, 2, 1)                              # (B, 27, C)
        x_t = x_t + self.pos_embed                            # add position embedding
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


def _build_enc_proj(raw_dim: int, enc_out_dim: int, num_layers: int = 1) -> nn.Sequential:
    """构建 enc_proj 投影层。

    Args:
        raw_dim: 展平后的输入维度 (conv_ch * NUM_TILES, 如 6912)
        enc_out_dim: 最终输出维度 (如 1024)
        num_layers: 投影层数
            1 = 单层 Linear（旧行为，向后兼容）
            2 = 渐进压缩 MLP (LayerNorm + Mish)，中间维度 = enc_out_dim * 2
                将单步高压缩比拆分为两步渐进，缓解信息瓶颈
    """
    if num_layers == 1:
        # 旧行为：LayerNorm + 单层 Linear
        return nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, enc_out_dim),
        )
    elif num_layers == 2:
        # 渐进压缩：raw_dim → mid_dim → enc_out_dim
        # 中间维度自动计算为 enc_out_dim * 2，确保不超过 raw_dim
        mid_dim = min(enc_out_dim * 2, raw_dim)
        return nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, mid_dim),          # 第一步压缩
            nn.LayerNorm(mid_dim),                # 中间归一化（用 LayerNorm 适配 2D 输入）
            nn.Mish(inplace=True),                # 与 ResBlock 保持一致的激活函数
            nn.Linear(mid_dim, enc_out_dim),      # 第二步压缩
        )
    else:
        raise ValueError(f"blood_enc_proj_layers 仅支持 1 或 2，当前值: {num_layers}")


class SuitAwareResNetEncoder(Encoder):
    """SuitAwareResNet encoder for Sample Factory with Bottleneck blocks.

    Architecture:
        obs (B, C*27) -> reshape (B, C, 27)
        -> stem: SuitAwareConv(C -> conv_ch) + GroupNorm + Mish
        -> pos_enc: SuitPositionalEncoding (rank-aware embeddings)
        -> for each segment i in [0, n_segments):
               segments[i]: BottleneckBlock x n_blocks_i
               tile_attns[i]: TileAttention (cross-suit interaction)
        -> flatten -> enc_proj: Linear(conv_ch*27 -> enc_out_dim)
        -> [B, enc_out_dim]

    The number of TileAttention layers (= segments) is controlled by
    blood_num_tile_attn_layers (default 2, supports 2-6+). Residual blocks
    are distributed evenly across segments with remainder blocks allocated
    to earlier segments.

    注意力头数通过 blood_tile_attn_heads 控制:
        4头(默认): 旧行为
        8头: 增强多模式跨花色交互

    enc_proj reduces the LSTM input from 6912 to enc_out_dim (default 1024),
    giving a 1:1 compression ratio inside the LSTM.

    当 enc_proj_layers=2 时，使用渐进压缩 MLP：
        6912 → mid_dim (enc_out_dim*2) → enc_out_dim
    将单步 6.75x 压缩拆分为 3.38x + 2x 两步渐进，缓解信息瓶颈。

    ⚠️ BREAKING CHANGE: This refactored loop-based segment architecture is
    incompatible with checkpoints from the old hardcoded 2/3-layer layout.
    Old attribute names (res_blocks_1, res_blocks_2, tile_attn_mid, etc.)
    no longer exist; they are replaced by segments[i] and tile_attns[i].
    """

    # Maximum allowed encoder output dimension. Prevents accidental LSTM
    # parameter explosion when enc_out_dim is set equal to raw_dim (conv_ch*27).
    _MAX_ENC_OUT_DIM = 2048

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        self.obs_channels = getattr(cfg, "blood_obs_channels", DEFAULT_OBS_CHANNELS)
        conv_ch = getattr(cfg, "blood_conv_channels", 256)
        num_blocks = getattr(cfg, "blood_num_res_blocks", 20)
        enc_out_dim = getattr(cfg, "blood_encoder_out_dim", 1024)
        enc_proj_layers = getattr(cfg, "blood_enc_proj_layers", 1)
        num_tile_attn_layers = getattr(cfg, "blood_num_tile_attn_layers", 2)
        tile_attn_heads = getattr(cfg, "blood_tile_attn_heads", 4)

        ng = _num_groups(conv_ch)

        self.stem = nn.Sequential(
            SuitAwareConv1d(self.obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)

        # Loop-based segment architecture: evenly distribute residual blocks
        # across n_segments, each followed by a TileAttention layer.
        n_segments = num_tile_attn_layers
        blocks_per_segment = num_blocks // n_segments
        remainder = num_blocks % n_segments

        self.segments = nn.ModuleList()
        self.tile_attns = nn.ModuleList()
        for i in range(n_segments):
            # Distribute remainder blocks to earlier segments
            n_blks = blocks_per_segment + (1 if i < remainder else 0)
            self.segments.append(nn.Sequential(
                *[BottleneckBlock(conv_ch) for _ in range(n_blks)]
            ))
            self.tile_attns.append(
                TileAttention(conv_ch, num_heads=tile_attn_heads)
            )

        raw_dim = conv_ch * NUM_TILES  # e.g. 256*27 = 6912
        # Guard: if enc_out_dim >= raw_dim, the projection would be a no-op or
        # expansion, and the LSTM input dimension would explode. Cap it.
        if enc_out_dim >= raw_dim:
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                "blood_encoder_out_dim (%d) >= raw_dim (%d); capping to %d "
                "to prevent LSTM parameter explosion.",
                enc_out_dim, raw_dim, self._MAX_ENC_OUT_DIM,
            )
            enc_out_dim = self._MAX_ENC_OUT_DIM

        self.enc_proj = _build_enc_proj(raw_dim, enc_out_dim, enc_proj_layers)
        self._out_size = enc_out_dim

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        B = obs.shape[0]
        x = obs.view(B, self.obs_channels, NUM_TILES)
        x = self.stem(x)
        x = self.pos_enc(x)
        for i in range(len(self.segments)):
            x = self.segments[i](x)
            x = self.tile_attns[i](x)
        flat = x.reshape(B, -1)
        flat = self.enc_proj(flat)
        return flat

    def get_out_size(self) -> int:
        return self._out_size
