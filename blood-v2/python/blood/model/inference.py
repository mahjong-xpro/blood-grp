"""Lightweight inference model for self-play opponent decisions.

Loads encoder + action head from an SF2 checkpoint and provides
a fast `get_action(obs, mask)` interface for use inside the environment.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from blood.model.encoder import (
    SuitAwareConv1d, BottleneckBlock, ChannelAttention,
    SuitPositionalEncoding, TileAttention,
    _num_groups, NUM_TILES, DEFAULT_OBS_CHANNELS,
)

log = logging.getLogger(__name__)

ACTION_DIM = 34


class PolicyModel(nn.Module):
    """Standalone policy model for opponent inference.

    Mirrors the SuitAwareResNetEncoder architecture (stem + pos_enc +
    res_blocks + tile_attn) paired with a simple action head.
    Weights are loaded from SF2 checkpoints.
    """

    def __init__(
        self,
        obs_channels: int = DEFAULT_OBS_CHANNELS,
        conv_ch: int = 256,
        num_blocks: int = 20,
        encoder_out: int = 1024,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()

        ng = _num_groups(conv_ch)
        self.stem = nn.Sequential(
            SuitAwareConv1d(obs_channels, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        self.pos_enc = SuitPositionalEncoding(conv_ch)
        self.res_blocks = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(num_blocks)])
        self.tile_attn = TileAttention(conv_ch, num_heads=4)

        flat_dim = conv_ch * NUM_TILES
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, encoder_out),
            nn.Mish(inplace=True),
        )

        self.action_head = nn.Linear(encoder_out, action_dim)
        self._obs_channels = obs_channels

    def forward(self, obs_flat: Tensor) -> Tensor:
        """obs_flat: (B, C*27) → logits (B, action_dim)"""
        B = obs_flat.shape[0]
        x = obs_flat.view(B, self._obs_channels, NUM_TILES)
        x = self.stem(x)
        x = self.pos_enc(x)
        x = self.res_blocks(x)
        x = self.tile_attn(x)
        features = self.fc(x.reshape(B, -1))
        return self.action_head(features)

    @torch.no_grad()
    def get_action(self, obs_flat: Tensor, mask: Tensor, temperature: float = 0.5) -> int:
        """Single-sample masked action selection with temperature sampling."""
        logits = self.forward(obs_flat.unsqueeze(0)).squeeze(0)
        logits[mask < 0.5] = -1e9
        if temperature <= 0.01:
            return int(logits.argmax().item())
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @classmethod
    def from_sf2_checkpoint(cls, path: str, device: str = "cpu") -> "PolicyModel":
        """Load from a Sample Factory 2 checkpoint.

        Extracts encoder weights and the action distribution head.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model_sd = ckpt.get("model", ckpt)

        encoder_sd = {}
        for k, v in model_sd.items():
            if k.startswith("encoder."):
                new_key = k[len("encoder."):]
                encoder_sd[new_key] = v

        action_weight = model_sd.get(
            "action_parameterization.distribution_linear.weight"
        )
        action_bias = model_sd.get(
            "action_parameterization.distribution_linear.bias"
        )

        obs_channels = DEFAULT_OBS_CHANNELS
        conv_ch = 256
        encoder_out = 1024

        # Detect obs_channels and conv_ch from stem key
        first_conv_key = "stem.0.conv.weight"
        if first_conv_key in encoder_sd:
            obs_channels = encoder_sd[first_conv_key].shape[1]
            conv_ch = encoder_sd[first_conv_key].shape[0]

        # Detect encoder_out from fc projection key
        fc_key = "fc.0.weight"
        if fc_key in encoder_sd:
            encoder_out = encoder_sd[fc_key].shape[0]

        # Detect num_blocks from res_blocks keys
        num_blocks = 0
        while f"res_blocks.{num_blocks}.block.0.weight" in encoder_sd:
            num_blocks += 1
        if num_blocks == 0:
            log.warning(
                "Could not detect num_blocks from checkpoint keys; defaulting to 20. "
                "First encoder keys: %s",
                [k for k in encoder_sd][:5],
            )
            num_blocks = 20

        model = cls(
            obs_channels=obs_channels,
            conv_ch=conv_ch,
            num_blocks=num_blocks,
            encoder_out=encoder_out,
        )
        model.to(device)

        partial_sd = {}
        for k, v in encoder_sd.items():
            if k in model.state_dict():
                partial_sd[k] = v

        if action_weight is not None:
            partial_sd["action_head.weight"] = action_weight
        if action_bias is not None:
            partial_sd["action_head.bias"] = action_bias

        model.load_state_dict(partial_sd, strict=False)
        model.eval()
        log.info("Loaded opponent model from %s (%d keys)", path, len(partial_sd))
        return model


class OpponentModelPool:
    """Manages a cached opponent model loaded from the league pool."""

    def __init__(self, device: str = "cpu", temperature: float = 0.5):
        self._model: Optional[PolicyModel] = None
        self._current_path: Optional[str] = None
        self._device = device
        self._temperature = temperature

    @property
    def ready(self) -> bool:
        return self._model is not None

    def load(self, path: str) -> None:
        path_str = str(path)
        if path_str == self._current_path:
            return
        try:
            self._model = PolicyModel.from_sf2_checkpoint(path_str, self._device)
            self._current_path = path_str
        except Exception as e:
            log.warning("Failed to load opponent model from %s: %s", path, e)

    @torch.no_grad()
    def get_action(self, obs: Tensor, mask: Tensor, temperature: float = None) -> int:
        if self._model is None:
            return self._fallback_action(mask)
        temp = temperature if temperature is not None else self._temperature
        return self._model.get_action(obs, mask, temperature=temp)

    @staticmethod
    def _fallback_action(mask: Tensor) -> int:
        """Random legal action when no model is available."""
        legal = (mask > 0.5).nonzero(as_tuple=True)[0]
        if len(legal) == 0:
            return 30  # Pass
        idx = torch.randint(0, len(legal), (1,)).item()
        return int(legal[idx].item())
