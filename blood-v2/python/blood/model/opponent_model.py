"""Lightweight opponent hand prediction network.

Predicts P(tile_t ∈ opponent_hand) for each tile type from public information.
Trained with BCE loss using Oracle observations as ground truth labels.
Used at inference time to improve ISMCE constrained sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from blood.model.encoder import (
    SuitAwareConv1d, BottleneckBlock, TileAttention, _num_groups,
)

# Input channels for opponent prediction:
# - Opponent kawa (58 ch per opponent)
# - Opponent melds (8 ch)
# - Visible tiles (4 ch)
# - Game context (5 ch: wall remaining, turn progress, ding_que, etc.)
# Total: 75 ch per opponent
OPP_PRED_IN_CHANNELS = 75


class OpponentHandPredictor(nn.Module):
    """Lightweight opponent hand inference network.

    From public information, predicts the probability that each tile type
    is in a specific opponent's hand. Uses a small SuitAwareResNet backbone
    to maintain suit-awareness.

    Input:  (B, 75, 27) — opponent-specific public features
    Output: (B, 27) — P(tile_t in opponent hand) per tile type
    """

    def __init__(
        self,
        in_channels: int = OPP_PRED_IN_CHANNELS,
        conv_ch: int = 128,
        num_blocks: int = 6,
        num_tile_attn: int = 1,
        tile_attn_heads: int = 4,
    ):
        super().__init__()
        ng = _num_groups(conv_ch)

        self.stem = nn.Sequential(
            SuitAwareConv1d(in_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(),
        )

        self.blocks = nn.Sequential(
            *[BottleneckBlock(conv_ch) for _ in range(num_blocks)]
        )

        self.tile_attn = nn.ModuleList([
            TileAttention(conv_ch, num_heads=tile_attn_heads)
            for _ in range(num_tile_attn)
        ])

        # Per-tile prediction head (preserves spatial dimension)
        self.head = nn.Sequential(
            nn.GroupNorm(ng, conv_ch),
            nn.Conv1d(conv_ch, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, 27) opponent-specific public features
        Returns:
            (B, 27) probabilities per tile type
        """
        x = self.stem(x)
        x = self.blocks(x)
        for attn in self.tile_attn:
            x = attn(x)
        # Return raw logits (pre-sigmoid) for numerically stable loss computation
        return self.head(x).squeeze(1)  # (B, 27)

    def predict_probs(self, x: Tensor) -> Tensor:
        """Return probabilities (sigmoid applied). Use for inference only."""
        return torch.sigmoid(self.forward(x))

    def loss(self, logits: Tensor, target: Tensor) -> Tensor:
        """Numerically stable BCE loss using logits directly.

        Args:
            logits: (B, 27) raw logits (pre-sigmoid) from forward()
            target: (B, 27) ground truth (from Oracle obs)
        """
        return F.binary_cross_entropy_with_logits(logits, target.clamp(0, 1))
