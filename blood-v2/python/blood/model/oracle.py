"""Oracle encoder and distillation loss.

The Oracle sees perfect information (all hands + wall) and is trained
jointly with the student. KL divergence from oracle to student policy
provides the distillation signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from blood.consts import NUM_ORACLE_CHANNELS
from blood.model.encoder import (
    SuitAwareConv1d, BottleneckBlock, SuitPositionalEncoding, TileAttention,
    NUM_TILES, _num_groups,
)

DEFAULT_ORACLE_CHANNELS = NUM_ORACLE_CHANNELS


class OracleEncoder(nn.Module):
    """Oracle encoder: same loop-based segment architecture as student but with perfect info.

    Uses the same generalized segment design as SuitAwareResNetEncoder:
    residual blocks are evenly distributed across n_segments, each followed
    by a TileAttention layer.

    ⚠️ BREAKING CHANGE: This refactored loop-based segment architecture is
    incompatible with checkpoints from the old hardcoded 2-layer layout.
    Old attribute names (res_blocks_1, res_blocks_2, tile_attn_mid, tile_attn)
    no longer exist; they are replaced by segments[i] and tile_attns[i].
    """

    def __init__(
        self,
        obs_channels: int = DEFAULT_ORACLE_CHANNELS,
        conv_ch: int = 256,
        num_blocks: int = 20,
        action_dim: int = 34,
        num_tile_attn_layers: int = 2,
        tile_attn_heads: int = 4,
    ):
        super().__init__()

        ng = _num_groups(conv_ch)
        self.stem = nn.Sequential(
            SuitAwareConv1d(obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)

        # Loop-based segment architecture (mirrors SuitAwareResNetEncoder)
        n_segments = num_tile_attn_layers
        blocks_per_segment = num_blocks // n_segments
        remainder = num_blocks % n_segments

        self.segments = nn.ModuleList()
        self.tile_attns = nn.ModuleList()
        for i in range(n_segments):
            n_blks = blocks_per_segment + (1 if i < remainder else 0)
            self.segments.append(nn.Sequential(
                *[BottleneckBlock(conv_ch) for _ in range(n_blks)]
            ))
            self.tile_attns.append(
                TileAttention(conv_ch, num_heads=tile_attn_heads)
            )

        # Output Trunk
        trunk_dim = conv_ch * NUM_TILES

        # Policy Head — Pre-norm 2-layer MLP matching student actor_head
        head_dim = 512
        self.policy_head = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, head_dim),
            nn.Mish(),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(),
            nn.Linear(head_dim, action_dim),
        )

        # Value Head — Pre-norm 2-layer MLP matching student critic_head.
        # Oracle has perfect information (all hands + wall), so its value
        # estimate is far more accurate than the student's partial-info estimate.
        # Oracle value distillation trains the student's critic to match this
        # better-calibrated target, improving credit assignment throughout training.
        self.value_head = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, head_dim),
            nn.Mish(),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, oracle_obs):
        """Returns (logits, values) — both computed from perfect-info observation."""
        B = oracle_obs.shape[0]
        x = oracle_obs.view(B, -1, NUM_TILES)
        x = self.stem(x)
        x = self.pos_enc(x)
        for i in range(len(self.segments)):
            x = self.segments[i](x)
            x = self.tile_attns[i](x)
        trunk = x.reshape(B, -1)
        logits = self.policy_head(trunk)
        values = self.value_head(trunk)
        return logits, values


class DistillationLoss(nn.Module):
    """KL divergence distillation from oracle to student."""

    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits, oracle_logits, action_mask=None):
        T = self.temperature

        if action_mask is not None:
            # Use -1e4 for float16 (safe under division by T >= 1.0) or -1e9 for float32.
            # torch.finfo(dtype).min is dangerous: dividing by T < 1.0 overflows float16,
            # and if all actions are masked, softmax produces 0/0 = NaN.
            if student_logits.dtype == torch.float16:
                large_neg = -1e4
            else:
                large_neg = -1e9
            student_logits = student_logits.masked_fill(~action_mask, large_neg)
            oracle_logits = oracle_logits.masked_fill(~action_mask, large_neg)

        # Use log_target=True for numerical stability: avoids computing oracle_probs
        # which can contain exact 0.0 in float16 (causing 0 * log(0) = NaN in kl_div).
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        oracle_log_probs = F.log_softmax(oracle_logits / T, dim=-1)

        kl = F.kl_div(student_log_probs, oracle_log_probs, reduction="batchmean", log_target=True)
        return kl * (T * T)
