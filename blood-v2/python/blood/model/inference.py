"""Lightweight inference model for self-play opponent decisions.

Loads encoder + LSTM + action head from an SF2 checkpoint and provides
a stateful `get_action(obs, mask, hidden_state)` interface for use inside
the environment. The LSTM hidden state is maintained per-opponent across
turns, matching the temporal modeling used during training.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from blood.model.encoder import (
    SuitAwareConv1d, BottleneckBlock, ChannelAttention,
    SuitPositionalEncoding, TileAttention,
    _num_groups, _build_enc_proj, NUM_TILES, DEFAULT_OBS_CHANNELS,
)

log = logging.getLogger(__name__)

ACTION_DIM = 34
HiddenState = Tuple[Tensor, Tensor]  # (h, c) for LSTM


class PolicyModel(nn.Module):
    """Standalone policy model for opponent inference.

    Mirrors the full training architecture (loop-based segments):
        encoder (stem + pos_enc + [segments[i] + tile_attns[i]] * n_segments
                 + enc_proj)
        → LSTM (temporal modeling across turns, 支持多层)
        → actor_head (2-layer Pre-norm MLP)
        → action_head (logits)

    Maintains LSTM hidden state across calls for temporal consistency.
    Weights are loaded from SF2 checkpoints via from_sf2_checkpoint().
    """

    def __init__(
        self,
        obs_channels: int = DEFAULT_OBS_CHANNELS,
        conv_ch: int = 256,
        num_blocks: int = 20,
        rnn_size: int = 1024,
        action_dim: int = ACTION_DIM,
        enc_out_dim: int = 1024,
        head_dim: int = 512,
        enc_proj_layers: int = 3,
        num_tile_attn_layers: int = 4,
        tile_attn_heads: int = 4,
        rnn_num_layers: int = 1,
    ):
        super().__init__()

        ng = _num_groups(conv_ch)

        # Stem
        self.stem = nn.Sequential(
            SuitAwareConv1d(obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)

        # Loop-based segment architecture matching encoder.py:
        # evenly distribute residual blocks across n_segments,
        # each followed by a TileAttention layer.
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

        raw_dim = conv_ch * NUM_TILES
        # enc_proj 使用与训练编码器相同的构建函数，支持渐进压缩
        self._use_spatial = (enc_proj_layers == 3)
        self.enc_proj = _build_enc_proj(raw_dim, enc_out_dim, enc_proj_layers, conv_ch=conv_ch)

        # LSTM 支持多层 (rnn_num_layers >= 1)
        self._rnn_num_layers = rnn_num_layers
        self.lstm = nn.LSTM(enc_out_dim, rnn_size, num_layers=rnn_num_layers, batch_first=True)
        self._rnn_size = rnn_size

        # 2-layer Pre-norm actor head matching training BloodActorCritic.actor_head
        self.actor_head = nn.Sequential(
            nn.LayerNorm(rnn_size),
            nn.Linear(rnn_size, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(inplace=True),
        )
        self.action_head = nn.Linear(head_dim, action_dim)
        self._obs_channels = obs_channels

    def _encode(self, obs_flat: Tensor) -> Tensor:
        B = obs_flat.shape[0]
        x = obs_flat.view(B, self._obs_channels, NUM_TILES)
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

    def forward(
        self,
        obs_flat: Tensor,
        hidden_state: Optional[HiddenState] = None,
    ) -> Tuple[Tensor, HiddenState]:
        """obs_flat: (B, C*27) → (logits (B, action_dim), new_hidden_state)"""
        enc = self._encode(obs_flat)
        lstm_out, new_hidden = self.lstm(enc.unsqueeze(1), hidden_state)
        features = self.actor_head(lstm_out.squeeze(1))
        return self.action_head(features), new_hidden

    @torch.no_grad()
    def get_action(
        self,
        obs_flat: Tensor,
        mask: Tensor,
        hidden_state: Optional[HiddenState] = None,
        temperature: float = 0.5,
    ) -> Tuple[int, HiddenState]:
        """Returns (action, new_hidden_state)."""
        logits, new_hidden = self.forward(obs_flat.unsqueeze(0), hidden_state)
        logits = logits.squeeze(0)
        # Use dtype-safe minimum instead of -1e9 to avoid overflow in float16
        logits[mask < 0.5] = torch.finfo(logits.dtype).min
        if temperature <= 0.01:
            return int(logits.argmax().item()), new_hidden
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item()), new_hidden

    def init_hidden(self, device: str = "cpu") -> HiddenState:
        # 多层 LSTM: hidden state shape = (num_layers, batch, rnn_size)
        h = torch.zeros(self._rnn_num_layers, 1, self._rnn_size, device=device)
        c = torch.zeros(self._rnn_num_layers, 1, self._rnn_size, device=device)
        return (h, c)

    @classmethod
    def from_sf2_checkpoint(cls, path: str, device: str = "cpu") -> "PolicyModel":
        """Load encoder + LSTM + actor_head + action_head from a Sample Factory 2 checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model_sd = ckpt.get("model", ckpt)

        encoder_sd = {k[len("encoder."):]: v for k, v in model_sd.items() if k.startswith("encoder.")}

        obs_channels = DEFAULT_OBS_CHANNELS
        conv_ch = 256
        rnn_size = 1024
        enc_out_dim = 1024

        first_conv_key = "stem.0.conv.weight"
        if first_conv_key in encoder_sd:
            obs_channels = encoder_sd[first_conv_key].shape[1]
            conv_ch = encoder_sd[first_conv_key].shape[0]

        # Detect rnn_size from LSTM hidden-hidden weight shape.
        # SF2 ModelCoreRNN stores the LSTM as self.core, so keys are core.core.*
        # (not core.rnn.* as one might expect).
        # Fallback: infer from actor_head.0 (LayerNorm weight shape = [rnn_size]).
        lstm_hh_key = "core.core.weight_hh_l0"
        if lstm_hh_key in model_sd:
            rnn_size = model_sd[lstm_hh_key].shape[1]
        elif "core.rnn.weight_hh_l0" in model_sd:  # legacy fallback
            rnn_size = model_sd["core.rnn.weight_hh_l0"].shape[1]
        elif "actor_head.0.weight" in model_sd:
            rnn_size = model_sd["actor_head.0.weight"].shape[0]

        # 检测 LSTM 层数: 检查 weight_hh_l1 是否存在来判断是否为多层
        rnn_num_layers = 1
        for prefix in ["core.core.", "core.rnn."]:
            layer_idx = 1
            while f"{prefix}weight_hh_l{layer_idx}" in model_sd:
                layer_idx += 1
            if layer_idx > 1:
                rnn_num_layers = layer_idx
                break

        # 检测 enc_proj 层数和 enc_out_dim
        # 3层 SpatialPoolingProj: enc_proj.queries 是独有的 Parameter
        # 2层渐进压缩格式: enc_proj = [LayerNorm(0), Linear(1), LayerNorm(2), Mish(3), Linear(4)]
        # 1层旧格式:       enc_proj = [LayerNorm(0), Linear(1)]
        enc_proj_layers = 1
        enc_proj_spatial_key = "enc_proj.queries"  # SpatialPoolingProj 独有
        enc_proj_2layer_key = "enc_proj.4.weight"  # 第二个 Linear 的权重
        enc_proj_1layer_key = "enc_proj.1.weight"  # 第一个（或唯一的）Linear 的权重
        if enc_proj_spatial_key in encoder_sd:
            # SpatialPoolingProj: enc_out_dim 从 proj Sequential 的 Linear 获取
            enc_proj_layers = 3
            enc_out_dim = encoder_sd["enc_proj.proj.1.weight"].shape[0]
        elif enc_proj_2layer_key in encoder_sd:
            # 2层渐进压缩：最终输出维度从第二个 Linear 获取
            enc_proj_layers = 2
            enc_out_dim = encoder_sd[enc_proj_2layer_key].shape[0]
        elif enc_proj_1layer_key in encoder_sd:
            enc_out_dim = encoder_sd[enc_proj_1layer_key].shape[0]
        else:
            enc_out_dim = conv_ch * NUM_TILES

        # Detect number of segments and total blocks from loop-based architecture.
        # New format: segments.0.0.block.0.weight, segments.1.0.block.0.weight, ...
        # Also support legacy format: res_blocks_1.0.block.0.weight, etc.
        num_tile_attn_layers = 0
        num_blocks = 0
        is_legacy = False

        # Try new loop-based format first
        seg_idx = 0
        while f"segments.{seg_idx}.0.block.0.weight" in encoder_sd:
            blk_idx = 0
            while f"segments.{seg_idx}.{blk_idx}.block.0.weight" in encoder_sd:
                blk_idx += 1
            num_blocks += blk_idx
            seg_idx += 1
        num_tile_attn_layers = seg_idx

        if num_tile_attn_layers == 0:
            # Fallback to legacy hardcoded format
            is_legacy = True
            num_blocks_1 = 0
            while f"res_blocks_1.{num_blocks_1}.block.0.weight" in encoder_sd:
                num_blocks_1 += 1

            num_tile_attn_layers = 2
            num_blocks_2 = 0
            if "res_blocks_2a.0.block.0.weight" in encoder_sd:
                # 3层模式: res_blocks_2a + tile_attn_mid2 + res_blocks_2b
                num_tile_attn_layers = 3
                num_blocks_2a = 0
                while f"res_blocks_2a.{num_blocks_2a}.block.0.weight" in encoder_sd:
                    num_blocks_2a += 1
                num_blocks_2b = 0
                while f"res_blocks_2b.{num_blocks_2b}.block.0.weight" in encoder_sd:
                    num_blocks_2b += 1
                num_blocks_2 = num_blocks_2a + num_blocks_2b
            else:
                while f"res_blocks_2.{num_blocks_2}.block.0.weight" in encoder_sd:
                    num_blocks_2 += 1

            num_blocks = num_blocks_1 + num_blocks_2

        if num_blocks == 0:
            log.warning("Could not detect num_blocks; defaulting to 20")
            num_blocks = 20
        if num_tile_attn_layers == 0:
            log.warning("Could not detect num_tile_attn_layers; defaulting to 2")
            num_tile_attn_layers = 2

        # 检测 TileAttention heads 数: 从 tile_attns.0.attn.in_proj_weight 推断
        tile_attn_heads = 4
        attn_proj_key = "tile_attns.0.attn.in_proj_weight"
        if attn_proj_key not in encoder_sd:
            # Legacy fallback
            attn_proj_key = "tile_attn.attn.in_proj_weight"
        # num_heads 不影响权重形状，使用默认值4

        # Detect head_dim from actor_head.4.weight (Pre-norm 2-layer)
        head_dim = 512
        if "actor_head.4.weight" in model_sd:
            head_dim = model_sd["actor_head.4.weight"].shape[0]

        model = cls(
            obs_channels=obs_channels,
            conv_ch=conv_ch,
            num_blocks=num_blocks,
            rnn_size=rnn_size,
            enc_out_dim=enc_out_dim,
            head_dim=head_dim,
            enc_proj_layers=enc_proj_layers,
            num_tile_attn_layers=num_tile_attn_layers,
            tile_attn_heads=tile_attn_heads,
            rnn_num_layers=rnn_num_layers,
        )
        model.to(device)

        partial_sd = {}

        if is_legacy:
            # Map legacy hardcoded keys to new loop-based keys.
            # Reconstruct how blocks were distributed in the old layout:
            #   res_blocks_1 had `mid` blocks, res_blocks_2 had the rest.
            #   For 3-layer: res_blocks_2a + res_blocks_2b split the second half.
            # We need to map these to segments[i] in the new layout.
            partial_sd.update(
                _map_legacy_encoder_to_segments(encoder_sd, model, num_tile_attn_layers)
            )
        else:
            # New format: encoder keys map directly (strip encoder. prefix already done)
            for k, v in encoder_sd.items():
                if k in model.state_dict():
                    partial_sd[k] = v

        # Map core.core.* → lstm.*  (SF2 ModelCoreRNN stores LSTM as self.core)
        # Also handle legacy core.rnn.* prefix for older checkpoints.
        # 支持多层 LSTM: weight_hh_l0, weight_hh_l1, ... 自动映射
        for k, v in model_sd.items():
            if k.startswith("core.core."):
                new_key = "lstm." + k[len("core.core."):]
                if new_key in model.state_dict():
                    partial_sd[new_key] = v
            elif k.startswith("core.rnn."):
                new_key = "lstm." + k[len("core.rnn."):]
                if new_key in model.state_dict():
                    partial_sd[new_key] = v

        # Load full actor_head weights (Pre-norm 2-layer MLP)
        for k, v in model_sd.items():
            if k.startswith("actor_head."):
                if k in model.state_dict():
                    partial_sd[k] = v

        # action_head from action_parameterization
        action_weight = model_sd.get("action_parameterization.distribution_linear.weight")
        action_bias = model_sd.get("action_parameterization.distribution_linear.bias")
        if action_weight is not None:
            partial_sd["action_head.weight"] = action_weight
        if action_bias is not None:
            partial_sd["action_head.bias"] = action_bias

        model.load_state_dict(partial_sd, strict=False)
        model.eval()
        log.info(
            "Loaded opponent model from %s (%d keys, rnn_size=%d, rnn_layers=%d, "
            "enc_out=%d, enc_proj_layers=%d, tile_attn_layers=%d, legacy=%s)",
            path, len(partial_sd), rnn_size, rnn_num_layers,
            enc_out_dim, enc_proj_layers, num_tile_attn_layers, is_legacy,
        )
        return model


def _map_legacy_encoder_to_segments(
    encoder_sd: dict,
    model: PolicyModel,
    num_tile_attn_layers: int,
) -> dict:
    """Map legacy hardcoded encoder keys to the new loop-based segment layout.

    Legacy layout (2-layer):
        res_blocks_1.{i} → segment 0, block i
        tile_attn_mid     → tile_attns.0
        res_blocks_2.{i}  → segment 1, block i
        tile_attn          → tile_attns.1

    Legacy layout (3-layer):
        res_blocks_1.{i}  → segment 0, block i
        tile_attn_mid      → tile_attns.0
        res_blocks_2a.{i}  → segment 1, block i
        tile_attn_mid2     → tile_attns.1
        res_blocks_2b.{i}  → segment 2, block i
        tile_attn           → tile_attns.2
    """
    mapped = {}
    model_sd = model.state_dict()

    # Map stem and pos_enc directly (same keys)
    for k, v in encoder_sd.items():
        if k.startswith("stem.") or k.startswith("pos_enc.") or k.startswith("enc_proj."):
            if k in model_sd:
                mapped[k] = v

    if num_tile_attn_layers == 2:
        # res_blocks_1.{i}.* → segments.0.{i}.*
        for k, v in encoder_sd.items():
            if k.startswith("res_blocks_1."):
                new_key = "segments.0." + k[len("res_blocks_1."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # tile_attn_mid.* → tile_attns.0.*
        for k, v in encoder_sd.items():
            if k.startswith("tile_attn_mid."):
                new_key = "tile_attns.0." + k[len("tile_attn_mid."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # res_blocks_2.{i}.* → segments.1.{i}.*
        for k, v in encoder_sd.items():
            if k.startswith("res_blocks_2."):
                new_key = "segments.1." + k[len("res_blocks_2."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # tile_attn.* → tile_attns.1.*
        for k, v in encoder_sd.items():
            if k.startswith("tile_attn.") and not k.startswith("tile_attn_mid"):
                new_key = "tile_attns.1." + k[len("tile_attn."):]
                if new_key in model_sd:
                    mapped[new_key] = v

    elif num_tile_attn_layers == 3:
        # res_blocks_1.{i}.* → segments.0.{i}.*
        for k, v in encoder_sd.items():
            if k.startswith("res_blocks_1."):
                new_key = "segments.0." + k[len("res_blocks_1."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # tile_attn_mid.* → tile_attns.0.*
        for k, v in encoder_sd.items():
            if k.startswith("tile_attn_mid.") and not k.startswith("tile_attn_mid2"):
                new_key = "tile_attns.0." + k[len("tile_attn_mid."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # res_blocks_2a.{i}.* → segments.1.{i}.*
        for k, v in encoder_sd.items():
            if k.startswith("res_blocks_2a."):
                new_key = "segments.1." + k[len("res_blocks_2a."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # tile_attn_mid2.* → tile_attns.1.*
        for k, v in encoder_sd.items():
            if k.startswith("tile_attn_mid2."):
                new_key = "tile_attns.1." + k[len("tile_attn_mid2."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # res_blocks_2b.{i}.* → segments.2.{i}.*
        for k, v in encoder_sd.items():
            if k.startswith("res_blocks_2b."):
                new_key = "segments.2." + k[len("res_blocks_2b."):]
                if new_key in model_sd:
                    mapped[new_key] = v
        # tile_attn.* → tile_attns.2.*
        for k, v in encoder_sd.items():
            if k.startswith("tile_attn.") and not k.startswith("tile_attn_mid"):
                new_key = "tile_attns.2." + k[len("tile_attn."):]
                if new_key in model_sd:
                    mapped[new_key] = v

    return mapped


class OpponentModelPool:
    """Manages a cached opponent model with per-opponent LSTM hidden states.

    Hidden states are maintained across turns within an episode and reset
    at the start of each new episode, matching the training model's behavior.
    """

    def __init__(self, device: str = "cpu", temperature: float = 0.5):
        self._model: Optional[PolicyModel] = None
        self._current_path: Optional[str] = None
        self._device = device
        self._temperature = temperature
        # Per-opponent hidden states: {player_id: (h, c)}
        self._hidden_states: Dict[int, HiddenState] = {}

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self, path: str) -> bool:
        """Load a checkpoint. Returns True on success, False on failure."""
        path_str = str(path)
        if path_str == self._current_path:
            return True
        try:
            self._model = PolicyModel.from_sf2_checkpoint(path_str, self._device)
            self._current_path = path_str
            self._hidden_states.clear()
            log.info("Loaded opponent model: %s", path_str)
            return True
        except Exception as e:
            log.warning(
                "Failed to load opponent model from %s: %s. "
                "Falling back to %s.",
                path, e,
                "previous model" if self._model is not None else "random policy",
            )
            return False

    def reset_hidden_states(self) -> None:
        """Reset all opponent hidden states at episode boundaries."""
        self._hidden_states.clear()

    @torch.no_grad()
    def get_action(
        self,
        obs: Tensor,
        mask: Tensor,
        opponent_id: int = 0,
        temperature: float = None,
    ) -> int:
        # Fallback: no model loaded yet (league pool empty at training start)
        # → uniform random over legal actions, equivalent to a random policy.
        if self._model is None:
            return self._fallback_action(mask)
        temp = temperature if temperature is not None else self._temperature
        hidden = self._hidden_states.get(opponent_id)
        action, new_hidden = self._model.get_action(obs, mask, hidden, temperature=temp)
        self._hidden_states[opponent_id] = new_hidden
        return action

    @staticmethod
    def _fallback_action(mask: Tensor) -> int:
        legal = (mask > 0.5).nonzero(as_tuple=True)[0]
        if len(legal) == 0:
            return 30  # Pass
        idx = torch.randint(0, len(legal), (1,)).item()
        return int(legal[idx].item())
