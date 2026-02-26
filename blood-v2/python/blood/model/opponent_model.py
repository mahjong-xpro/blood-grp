"""Lightweight opponent hand prediction network.

Predicts P(tile_t ∈ opponent_hand) for each tile type from public information.
Trained with BCE loss using Oracle observations as ground truth labels.
Used at inference time to improve ISMCE constrained sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from blood.model.encoder import (
    SuitAwareConv1d, BottleneckBlock, TileAttention, _num_groups,
)
from blood.consts import (
    NUM_TILE_TYPES, NUM_STUDENT_CHANNELS,
    CH_OPP_KAWA_BASE, CH_OPP_KAWA_STRIDE,
    CH_VISIBLE_TILES_BASE, CH_WALL_REMAINING, CH_TURN_PROGRESS,
    CH_OPP_DING_QUE_BASE,
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

    @staticmethod
    def build_features_numpy(obs_2d: np.ndarray, opp_idx: int) -> np.ndarray:
        """Build 75ch input features for a single opponent from student obs.

        Fix R11-H4: single source of truth for feature construction, used by
        both losses.py (training) and ismce.py (inference).

        Args:
            obs_2d: (C, 27) student observation reshaped to 2D (numpy)
            opp_idx: opponent index (0, 1, 2)
        Returns:
            (75, 27) numpy array of opponent-specific features
        """
        features = np.zeros((OPP_PRED_IN_CHANNELS, NUM_TILE_TYPES), dtype=np.float32)
        fuuro_base = CH_VISIBLE_TILES_BASE + 12  # skip 3×4 kawa overview
        opp_fuuro_base = fuuro_base + 8  # skip self fuuro (8ch)
        ch = 0

        # [0:58] Opponent kawa (58 ch)
        ks = CH_OPP_KAWA_BASE + opp_idx * CH_OPP_KAWA_STRIDE
        ke = ks + CH_OPP_KAWA_STRIDE
        if ke <= obs_2d.shape[0]:
            features[ch:ch + CH_OPP_KAWA_STRIDE] = obs_2d[ks:ke]
        ch += CH_OPP_KAWA_STRIDE

        # [58:62] Visible tiles / kawa overview (4 ch)
        vs = CH_VISIBLE_TILES_BASE + opp_idx * 4
        if vs + 4 <= obs_2d.shape[0]:
            features[ch:ch + 4] = obs_2d[vs:vs + 4]
        ch += 4

        # [62:70] Opponent fuuro (8 ch)
        ms = opp_fuuro_base + opp_idx * 8
        if ms + 8 <= obs_2d.shape[0]:
            features[ch:ch + 8] = obs_2d[ms:ms + 8]
        ch += 8

        # [70] Wall remaining (1 ch)
        if CH_WALL_REMAINING < obs_2d.shape[0]:
            features[ch] = obs_2d[CH_WALL_REMAINING]
        ch += 1

        # [71] Turn progress (1 ch)
        if CH_TURN_PROGRESS < obs_2d.shape[0]:
            features[ch] = obs_2d[CH_TURN_PROGRESS]
        ch += 1

        # [72:75] Opponent ding-que (3 ch)
        dqs = CH_OPP_DING_QUE_BASE + opp_idx * 3
        if dqs + 3 <= obs_2d.shape[0]:
            features[ch:ch + 3] = obs_2d[dqs:dqs + 3]

        return features

    @staticmethod
    def build_features_tensor(student_2d: Tensor, opp_idx: int) -> Tensor:
        """Build 75ch input features for a single opponent from student obs (batched).

        Args:
            student_2d: (B, C, 27) student observation tensor
            opp_idx: opponent index (0, 1, 2)
        Returns:
            (B, 75, 27) tensor of opponent-specific features
        """
        B = student_2d.shape[0]
        features = torch.zeros(B, OPP_PRED_IN_CHANNELS, NUM_TILE_TYPES,
                               device=student_2d.device, dtype=student_2d.dtype)
        fuuro_base = CH_VISIBLE_TILES_BASE + 12
        opp_fuuro_base = fuuro_base + 8
        ch = 0

        # [0:58] Opponent kawa
        ks = CH_OPP_KAWA_BASE + opp_idx * CH_OPP_KAWA_STRIDE
        ke = ks + CH_OPP_KAWA_STRIDE
        if ke <= student_2d.shape[1]:
            features[:, ch:ch + CH_OPP_KAWA_STRIDE, :] = student_2d[:, ks:ke, :]
        ch += CH_OPP_KAWA_STRIDE

        # [58:62] Visible tiles
        vs = CH_VISIBLE_TILES_BASE + opp_idx * 4
        if vs + 4 <= student_2d.shape[1]:
            features[:, ch:ch + 4, :] = student_2d[:, vs:vs + 4, :]
        ch += 4

        # [62:70] Fuuro
        ms = opp_fuuro_base + opp_idx * 8
        if ms + 8 <= student_2d.shape[1]:
            features[:, ch:ch + 8, :] = student_2d[:, ms:ms + 8, :]
        ch += 8

        # [70] Wall remaining
        if CH_WALL_REMAINING < student_2d.shape[1]:
            features[:, ch, :] = student_2d[:, CH_WALL_REMAINING, :]
        ch += 1

        # [71] Turn progress
        if CH_TURN_PROGRESS < student_2d.shape[1]:
            features[:, ch, :] = student_2d[:, CH_TURN_PROGRESS, :]
        ch += 1

        # [72:75] Ding-que
        dqs = CH_OPP_DING_QUE_BASE + opp_idx * 3
        if dqs + 3 <= student_2d.shape[1]:
            features[:, ch:ch + 3, :] = student_2d[:, dqs:dqs + 3, :]

        return features
