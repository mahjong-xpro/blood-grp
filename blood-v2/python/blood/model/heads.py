"""Auxiliary task heads for Bloody Battle Mahjong.

- DingQue prediction (3-class CE)
- Opponent waits prediction (81-dim BCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxHead(nn.Module):
    """Auxiliary prediction head: ding_que (3×3-class) + opponent_waits (81-dim BCE)."""

    NUM_OPPONENTS = 3
    NUM_SUITS = 3

    def __init__(self, in_dim: int = 1024, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Mish(inplace=True),
        )
        self.dq_head = nn.Linear(hidden, self.NUM_OPPONENTS * self.NUM_SUITS)
        self.ow_head = nn.Linear(hidden, self.NUM_OPPONENTS * 27)

    def forward(self, features: torch.Tensor):
        h = self.shared(features)
        dq_logits = self.dq_head(h).view(-1, self.NUM_OPPONENTS, self.NUM_SUITS)
        ow_logits = self.ow_head(h)
        return dq_logits, ow_logits

    def loss(self, features, dq_labels, ow_labels, dq_weight=1.0, ow_weight=0.1):
        dq_logits, ow_logits = self.forward(features)

        dq_loss = F.cross_entropy(
            dq_logits.reshape(-1, self.NUM_SUITS),
            dq_labels.reshape(-1),
            ignore_index=self.NUM_SUITS,
            reduction="mean",
        )

        if ow_weight > 0:
            ow_per_sample = F.binary_cross_entropy_with_logits(
                ow_logits, ow_labels, reduction="none",
            ).mean(dim=-1)
            ow_mask = ow_labels.abs().sum(dim=-1) > 0.01
            if ow_mask.any():
                ow_loss = ow_per_sample[ow_mask].mean()
            else:
                ow_loss = torch.tensor(0.0, device=features.device)
        else:
            ow_loss = torch.tensor(0.0, device=features.device)

        return dq_weight * dq_loss + ow_weight * ow_loss
