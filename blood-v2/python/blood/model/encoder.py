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
            nn.Mish(),
            nn.Conv1d(channels, mid_channels, 1, bias=False),  # 1x1 down
            nn.GroupNorm(mid_ng, mid_channels),
            nn.Mish(),
            SuitAwareConv1d(mid_channels, mid_channels, kernel_size=3),
            nn.GroupNorm(mid_ng, mid_channels),
            nn.Mish(),
            nn.Conv1d(mid_channels, channels, 1, bias=False),  # 1x1 up
        )
        self.attn = ChannelAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.block(x))


class ChannelAttention(nn.Module):
    """Squeeze-Excitation channel attention for cross-suit information exchange.

    Shares MLP weights between avg-pool and max-pool branches (standard CBAM/SE-Net
    design). Only the pooling operation differs; the shared MLP learns a single
    channel importance mapping applied to both pooled representations.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # Shared MLP across both pooling branches
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.Mish(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1)
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
            nn.Mish(),
            SuitAwareConv1d(channels, channels, kernel_size=3),
            nn.GroupNorm(ng, channels),
            nn.Mish(),
            SuitAwareConv1d(channels, channels, kernel_size=3),
        )
        self.attn = ChannelAttention(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.block(x))


class SpatialPoolingProj(nn.Module):
    """Attention-based spatial pooling to replace flatten + linear projection.

    Uses learnable query tokens with cross-attention to aggregate information
    from tile positions, preserving spatial structure instead of brute-force
    flattening.

    Architecture:
        Input: (B, conv_ch, 27)  -- tile-position features from ResNet encoder
        → LayerNorm over channels
        → MultiheadAttention(queries=learnable, keys/values=tile positions)
        → Flatten queries → (B, num_queries * conv_ch)
        → Linear → (B, enc_out_dim)

    With num_queries=4 and conv_ch=256: 4 queries attend over 27 positions,
    each producing a 256-dim summary → concat to 1024 → project to enc_out_dim.
    Compression is done by the attention mechanism which adaptively selects
    important tile positions, rather than a blind linear map.
    """

    def __init__(self, conv_ch: int, enc_out_dim: int, num_queries: int = 4,
                 num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.conv_ch = conv_ch
        self.num_queries = num_queries

        # Learnable query tokens
        self.queries = nn.Parameter(torch.zeros(1, num_queries, conv_ch))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Pre-norm for stable attention
        self.norm = nn.LayerNorm(conv_ch)
        self.query_norm = nn.LayerNorm(conv_ch)  # Symmetric normalization for queries

        # Cross-attention: queries attend to tile positions (keys/values)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=conv_ch,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Final projection from (num_queries * conv_ch) → enc_out_dim
        query_dim = num_queries * conv_ch
        self.proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, enc_out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, conv_ch, 27) — spatial tile features
        B = x.shape[0]
        # Transpose to (B, 27, conv_ch) for attention
        kv = x.permute(0, 2, 1)
        kv = self.norm(kv)

        # Expand queries for batch and normalize symmetrically with keys/values
        q = self.query_norm(self.queries.expand(B, -1, -1))  # (B, num_queries, conv_ch)

        # Cross-attention: queries attend to tile positions
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, num_queries, conv_ch)

        # Flatten and project
        flat = attn_out.reshape(B, -1)  # (B, num_queries * conv_ch)
        return self.proj(flat)  # (B, enc_out_dim)


def _build_enc_proj(raw_dim: int, enc_out_dim: int, num_layers: int = 1,
                    conv_ch: int = 256) -> nn.Module:
    """构建 enc_proj 投影层。

    Args:
        raw_dim: 展平后的输入维度 (conv_ch * NUM_TILES, 如 6912)
        enc_out_dim: 最终输出维度 (如 1024)
        num_layers: 投影层数
            1 = 单层 Linear（旧行为，向后兼容）
            2 = 渐进压缩 MLP (LayerNorm + Mish)，中间维度 = enc_out_dim * 2
                将单步高压缩比拆分为两步渐进，缓解信息瓶颈
            3 = SpatialPoolingProj（注意力池化，保留牌位结构，最少信息损失）
        conv_ch: 卷积通道数，仅 num_layers=3 时使用
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
    elif num_layers == 3:
        # 注意力池化：SpatialPoolingProj
        # 自动计算 num_queries = enc_out_dim // conv_ch (通常 1024/256 = 4)
        num_queries = max(enc_out_dim // conv_ch, 2)
        return SpatialPoolingProj(
            conv_ch=conv_ch,
            enc_out_dim=enc_out_dim,
            num_queries=num_queries,
        )
    else:
        raise ValueError(f"blood_enc_proj_layers 仅支持 1, 2 或 3，当前值: {num_layers}")


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
    blood_num_tile_attn_layers (default 4, supports 2-6+). Residual blocks
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
        enc_proj_layers = getattr(cfg, "blood_enc_proj_layers", 3)
        num_tile_attn_layers = getattr(cfg, "blood_num_tile_attn_layers", 4)
        tile_attn_heads = getattr(cfg, "blood_tile_attn_heads", 4)

        ng = _num_groups(conv_ch)

        self.stem = nn.Sequential(
            SuitAwareConv1d(self.obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(),
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

        self._use_spatial = (enc_proj_layers == 3)
        self.enc_proj = _build_enc_proj(raw_dim, enc_out_dim, enc_proj_layers, conv_ch=conv_ch)
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
        if self._use_spatial:
            # SpatialPoolingProj takes (B, C, 27) directly
            return self.enc_proj(x)
        else:
            flat = x.reshape(B, -1)
            return self.enc_proj(flat)

    def get_out_size(self) -> int:
        return self._out_size
