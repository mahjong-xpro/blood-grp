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
    """Oracle encoder: same architecture as student but with perfect info."""

    def __init__(
        self,
        obs_channels: int = DEFAULT_ORACLE_CHANNELS,
        conv_ch: int = 256,
        num_blocks: int = 20,
        action_dim: int = 34,
    ):
        super().__init__()

        ng = _num_groups(conv_ch)
        mid = num_blocks // 2
        self.stem = nn.Sequential(
            SuitAwareConv1d(obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)
        self.res_blocks_1 = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(mid)])
        self.tile_attn_mid = TileAttention(conv_ch, num_heads=4)
        self.res_blocks_2 = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(num_blocks - mid)])
        self.tile_attn = TileAttention(conv_ch, num_heads=4)

        # Output Trunk
        trunk_dim = conv_ch * NUM_TILES

        # Policy Head — Pre-norm 2-layer MLP matching student actor_head
        head_dim = 512
        self.policy_head = nn.Sequential(
            nn.LayerNorm(trunk_dim),
            nn.Linear(trunk_dim, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(inplace=True),
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
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(inplace=True),
            nn.Linear(head_dim, 1),
        )

    def forward(self, oracle_obs):
        """Returns (logits, values) — both computed from perfect-info observation."""
        B = oracle_obs.shape[0]
        x = oracle_obs.view(B, -1, NUM_TILES)
        x = self.stem(x)
        x = self.pos_enc(x)
        x = self.res_blocks_1(x)
        x = self.tile_attn_mid(x)
        x = self.res_blocks_2(x)
        x = self.tile_attn(x)
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
            large_neg = torch.finfo(student_logits.dtype).min
            student_logits = student_logits.masked_fill(~action_mask, large_neg)
            oracle_logits = oracle_logits.masked_fill(~action_mask, large_neg)

        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        oracle_probs = F.softmax(oracle_logits / T, dim=-1)

        kl = F.kl_div(student_log_probs, oracle_probs, reduction="batchmean")
        return kl * (T * T)
