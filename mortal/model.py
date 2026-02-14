import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import *
from functools import partial
from libblood.consts import obs_shape, oracle_obs_shape, ACTION_SPACE

TILE_KINDS = 27
SUIT_WIDTH = 9   # 每花色 9 种牌
NUM_SUITS = 3    # 万/筒/条


class SuitAwareConv1d(nn.Module):
    """Conv1d that isolates the three suits (Man/Pin/Sou, each 9 tiles)
    to prevent cross-suit boundary convolution.

    MODEL-04 fix: Standard Conv1d(kernel=3, padding=1) on 27-width input
    creates cross-suit contamination at positions 8↔9 (9m↔1p) and 17↔18
    (9p↔1s). This wrapper reshapes the input so each suit becomes a separate
    batch element, pads each suit independently, applies a single Conv1d call,
    and reshapes back. The kernel weights are shared across all three suits.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        assert kernel_size == 3 and padding == 1, \
            "SuitAwareConv1d only supports kernel_size=3, padding=1"
        # Internal conv uses padding=0; we handle padding manually per suit.
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        B, C, W = x.shape
        assert W == TILE_KINDS, f"Expected width {TILE_KINDS}, got {W}"
        # (B, C, 3, 9) → (B, 3, C, 9) → (B*3, C, 9)
        x3 = x.view(B, C, NUM_SUITS, SUIT_WIDTH).transpose(1, 2).contiguous().view(B * NUM_SUITS, C, SUIT_WIDTH)
        # Pad each suit independently: (B*3, C, 11)
        x3 = F.pad(x3, (1, 1))
        # Single conv call: (B*3, out_C, 9)
        out = self.conv(x3)
        out_C = out.shape[1]
        # (B*3, out_C, 9) → (B, 3, out_C, 9) → (B, out_C, 3, 9) → (B, out_C, 27)
        out = out.view(B, NUM_SUITS, out_C, SUIT_WIDTH).transpose(1, 2).contiguous().view(B, out_C, TILE_KINDS)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16, actv_builder=nn.ReLU, bias=True):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=bias),
            actv_builder(),
            nn.Linear(channels // ratio, channels, bias=bias),
        )
        if bias:
            for mod in self.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.constant_(mod.bias, 0)

    def forward(self, x: Tensor):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        weight = (avg_out + max_out).sigmoid()
        x = weight.unsqueeze(-1) * x
        return x

class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()
        self.pre_actv = pre_actv

        if pre_actv:
            self.res_unit = nn.Sequential(
                norm_builder(),
                actv_builder(),
                SuitAwareConv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                SuitAwareConv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            )
        else:
            self.res_unit = nn.Sequential(
                SuitAwareConv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
                actv_builder(),
                SuitAwareConv1d(channels, channels, kernel_size=3, padding=1, bias=False),
                norm_builder(),
            )
            self.actv = actv_builder()
        self.ca = ChannelAttention(channels, actv_builder=actv_builder, bias=True)

    def forward(self, x):
        out = self.res_unit(x)
        out = self.ca(out)
        out = out + x
        if not self.pre_actv:
            out = self.actv(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_channels,
        num_blocks,
        *,
        norm_builder = nn.Identity,
        actv_builder = nn.ReLU,
        pre_actv = False,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(
                conv_channels,
                norm_builder = norm_builder,
                actv_builder = actv_builder,
                pre_actv = pre_actv,
            ))

        layers = [SuitAwareConv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False)]
        if pre_actv:
            layers += [*blocks, norm_builder(), actv_builder()]
        else:
            layers += [norm_builder(), actv_builder(), *blocks]
        # MODEL-06 fix: 最终 conv 通道 32→64，减轻瓶颈。
        # 旧: 192→32 (6x 压缩), flatten 32×27=864 → Linear(864, 1024)
        # 新: 192→64 (3x 压缩), flatten 64×27=1728 → Linear(1728, 1024)
        # 保留更多特征供 DQN V/A 流和 AuxNet 使用。
        layers += [
            SuitAwareConv1d(conv_channels, 64, kernel_size=3, padding=1, bias=True),
            actv_builder(),
            nn.Flatten(),
            nn.Linear(64 * TILE_KINDS, 1024),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=4):
        super().__init__()
        self.is_oracle = is_oracle

        in_channels = obs_shape(version)[0]
        if is_oracle:
            in_channels += oracle_obs_shape(version)[0]

        norm_builder = partial(nn.BatchNorm1d, conv_channels, momentum=0.01, eps=1e-3)
        actv_builder = partial(nn.Mish, inplace=True)

        self.encoder = ResNet(
            in_channels = in_channels,
            conv_channels = conv_channels,
            num_blocks = num_blocks,
            norm_builder = norm_builder,
            actv_builder = actv_builder,
            pre_actv = True,
        )
        self.actv = actv_builder()

        # always use EMA or CMA when True
        self._freeze_bn = False

    def forward(self, obs: Tensor, invisible_obs: Optional[Tensor] = None) -> Tensor:
        if self.is_oracle:
            assert invisible_obs is not None
            obs = torch.cat((obs, invisible_obs), dim=1)
        phi = self.encoder(obs)
        return self.actv(phi)

    def train(self, mode=True):
        super().train(mode)
        if self._freeze_bn:
            for mod in self.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
                    # I don't think this benefits
                    # module.requires_grad_(False)
        return self

    def reset_running_stats(self):
        for mod in self.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats()

    def freeze_bn(self, value: bool):
        self._freeze_bn = value
        return self.train(self.training)

class AuxNet(nn.Module):
    # MODEL-02 fix: 加入 512 维隐藏层 + Mish 激活 + bias。
    # 旧版为无偏置单线性层 (1024→85)，辅助梯度信号穿透浅，
    # 对手听牌预测（需弃牌分析/副露推断）能力严重受限。
    def __init__(self, dims=None, hidden=512):
        super().__init__()
        self.dims = dims
        self.net = nn.Sequential(
            nn.Linear(1024, hidden),
            nn.Mish(inplace=True),
            nn.Linear(hidden, sum(dims)),
        )

    def forward(self, x):
        return self.net(x).split(self.dims, dim=-1)

class DQN(nn.Module):
    # MODEL-01 fix: 独立 V/A 流 + 隐藏层，恢复 Dueling DQN 的真正优势。
    # 旧版用单个 nn.Linear(1024, 35)，V 和 A 共享权重矩阵，无非线性分离，
    # 使 Dueling 架构退化为普通 DQN。
    # 新版 V/A 各有独立 512 维隐藏层 + Mish 激活，~1M 新增参数（相对 Brain 可忽略）。
    def __init__(self, *, version=4):
        super().__init__()
        self.v_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, 1),
        )
        self.a_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, ACTION_SPACE),
        )
        # 初始化：A 输出层权重和偏置为零，使初始 Q ≈ V（稳定训练启动）
        nn.init.zeros_(self.a_stream[-1].weight)
        nn.init.zeros_(self.a_stream[-1].bias)
        nn.init.zeros_(self.v_stream[-1].bias)

    def forward(self, phi, mask):
        v = self.v_stream(phi)
        a = self.a_stream(phi)

        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
        mask_sum = mask.sum(-1, keepdim=True)
        a_mean = a_sum / mask_sum.clamp(min=1)
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q


