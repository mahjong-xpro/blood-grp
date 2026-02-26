"""血战麻将辅助任务头。

- 对手向听数预测 (3 opponents x 5-class CE)
- 对手听牌预测 (81-dim Focal Loss)

定缺预测已移除：对手定缺在学生观测 Section 3 (channels 5-21) 中直接可观测，
作为辅助任务是冗余的。

向听数预测有实际价值：模型需从可观测信息（弃牌、副露、游戏上下文）推断
对手进度 (0-4 向听)，直接影响防守决策。

听牌预测改用 Focal Loss：听牌是极度不平衡任务（大部分时间大部分对手不听牌），
标准 BCE 容易退化为"全部预测不听"。Focal Loss 对难分类样本给予更大权重。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Sigmoid Focal Loss，用于解决类别不平衡问题。

    对容易分类的样本降低权重，对难分类样本（如实际听牌但预测不听）给予更大权重。

    Args:
        inputs: 未经 sigmoid 的 logits
        targets: 二值标签 (0 或 1)
        alpha: 正样本权重因子，默认 0.25
        gamma: 聚焦参数，gamma 越大越关注难分类样本，默认 2.0
        reduction: 'mean' | 'sum' | 'none'
    """
    p = torch.sigmoid(inputs)
    # 标准 BCE 部分
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # focal 调制因子: (1 - p_t)^gamma
    # p_t = p (当 target=1) 或 1-p (当 target=0)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    # alpha 平衡因子: alpha (当 target=1) 或 1-alpha (当 target=0)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class AuxHead(nn.Module):
    """辅助预测头: opp_shanten (3x5-class CE) + opp_waits (81-dim Focal Loss)。"""

    NUM_OPPONENTS = 3
    NUM_SHANTEN_CLASSES = 5  # 0, 1, 2, 3, 4+

    def __init__(
        self,
        in_dim: int = 1024,
        hidden: int = 512,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        # Focal Loss 超参数
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Pre-norm 2层共享主干，与 actor/critic head 深度匹配。
        # 两层允许共享表示在任务特定头分叉前解耦向听数和听牌特征。
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
        """计算辅助任务损失。

        Args:
            shanten_labels: (B, 15) flat 或 (B, 3, 5) one-hot float
            ow_labels:      (B, 81) float binary
        """
        shanten_logits, ow_logits = self.forward(features)

        # 归一化 shanten_labels 为 (B, 3, 5)
        B = features.shape[0]
        sl = shanten_labels.view(B, self.NUM_OPPONENTS, self.NUM_SHANTEN_CLASSES)

        # one-hot 转 class index 用于 CE
        shanten_targets = sl.argmax(dim=-1)  # (B, 3)
        # Fix R12-H3: mask out opponents with all-zero shanten labels (已和牌).
        # argmax on all-zero rows returns 0 (tenpai), injecting false labels.
        valid_shanten = sl.sum(dim=-1) > 0.01  # (B, 3) — True if label is valid
        if valid_shanten.any():
            shanten_loss = F.cross_entropy(
                shanten_logits.reshape(-1, self.NUM_SHANTEN_CLASSES)[valid_shanten.reshape(-1)],
                shanten_targets.reshape(-1)[valid_shanten.reshape(-1)],
                reduction="mean",
            )
        else:
            shanten_loss = torch.tensor(0.0, device=features.device)

        if ow_weight > 0:
            # 重塑为 (B, 3, 27) 以按对手独立处理。
            # 仅对实际听牌的对手计算损失（ow_labels 非零）。
            # 混合听牌和非听牌对手会引入噪声（非听牌对手的全零行）。
            ow_per_opp = ow_logits.view(-1, self.NUM_OPPONENTS, 27)
            ow_labels_3d = ow_labels.view(-1, self.NUM_OPPONENTS, 27)
            opp_tenpai_mask = ow_labels_3d.abs().sum(dim=-1) > 0.01  # (B, 3)
            if opp_tenpai_mask.any():
                # 使用 Focal Loss 替代标准 BCE
                # 听牌是极度不平衡任务，Focal Loss 对难分类样本给予更大权重，
                # 避免模型退化为"全部预测不听"
                ow_loss = sigmoid_focal_loss(
                    ow_per_opp[opp_tenpai_mask],
                    ow_labels_3d[opp_tenpai_mask],
                    alpha=self.focal_alpha,
                    gamma=self.focal_gamma,
                    reduction="mean",
                )
            else:
                ow_loss = torch.tensor(0.0, device=features.device)
        else:
            ow_loss = torch.tensor(0.0, device=features.device)

        return shanten_weight * shanten_loss + ow_weight * ow_loss
