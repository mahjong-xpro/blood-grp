"""Auxiliary task heads for Bloody Battle Mahjong.

- Opponent shanten prediction (3 opponents x 5-class CE)
- Opponent waits prediction (81-dim BCE)

DingQue prediction was removed: opponent dingque is directly observable
in Section 3 of the student observation (channels 5-21), making it
redundant as an auxiliary task.

Shanten prediction is non-trivial: the model must infer opponent progress
(0-4 shanten) from observable information (discards, melds, game context),
which directly informs defensive decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxHead(nn.Module):
    """Auxiliary prediction head: opp_shanten (3x5-class CE) + opp_waits (81-dim BCE)."""

    NUM_OPPONENTS = 3
    NUM_SHANTEN_CLASSES = 5  # 0, 1, 2, 3, 4+

    def __init__(self, in_dim: int = 1024, hidden: int = 512):
        super().__init__()
        # Pre-norm 2-layer shared trunk matching actor/critic head depth.
        # Two layers allow the shared representation to disentangle shanten
        # and wait-tile features before the task-specific heads split off.
        self.shared = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.Mish(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.Mish(inplace=True),
        )
        self.shanten_head = nn.Linear(hidden, self.NUM_OPPONENTS * self.NUM_SHANTEN_CLASSES)
        self.ow_head = nn.Linear(hidden, self.NUM_OPPONENTS * 27)

    def forward(self, features: torch.Tensor):
        h = self.shared(features)
        shanten_logits = self.shanten_head(h).view(-1, self.NUM_OPPONENTS, self.NUM_SHANTEN_CLASSES)
        ow_logits = self.ow_head(h)
        return shanten_logits, ow_logits

    def loss(self, features, shanten_labels, ow_labels, shanten_weight=1.0, ow_weight=0.1):
        """
        shanten_labels: (B, 15) flat or (B, 3, 5) one-hot float
        ow_labels:      (B, 81) float binary
        """
        shanten_logits, ow_logits = self.forward(features)

        # Normalise shanten_labels to (B, 3, 5) regardless of input shape
        B = features.shape[0]
        sl = shanten_labels.view(B, self.NUM_OPPONENTS, self.NUM_SHANTEN_CLASSES)

        # Convert one-hot to class index for CE
        shanten_targets = sl.argmax(dim=-1)  # (B, 3)
        shanten_loss = F.cross_entropy(
            shanten_logits.reshape(-1, self.NUM_SHANTEN_CLASSES),
            shanten_targets.reshape(-1),
            reduction="mean",
        )

        if ow_weight > 0:
            # Reshape to (B, 3, 27) to mask per opponent independently.
            # Only compute BCE for opponents that are actually tenpai (ow_labels non-zero).
            # Mixing tenpai and non-tenpai opponents in a single mask introduces noise
            # from the all-zero rows of non-tenpai opponents.
            ow_per_opp = ow_logits.view(-1, self.NUM_OPPONENTS, 27)
            ow_labels_3d = ow_labels.view(-1, self.NUM_OPPONENTS, 27)
            opp_tenpai_mask = ow_labels_3d.abs().sum(dim=-1) > 0.01  # (B, 3)
            if opp_tenpai_mask.any():
                ow_loss = F.binary_cross_entropy_with_logits(
                    ow_per_opp[opp_tenpai_mask],
                    ow_labels_3d[opp_tenpai_mask],
                    reduction="mean",
                )
            else:
                ow_loss = torch.tensor(0.0, device=features.device)
        else:
            ow_loss = torch.tensor(0.0, device=features.device)

        return shanten_weight * shanten_loss + ow_weight * ow_loss
