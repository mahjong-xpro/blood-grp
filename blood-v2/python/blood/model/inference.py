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
    _num_groups, NUM_TILES, DEFAULT_OBS_CHANNELS,
)

log = logging.getLogger(__name__)

ACTION_DIM = 34
HiddenState = Tuple[Tensor, Tensor]  # (h, c) for LSTM


class PolicyModel(nn.Module):
    """Standalone policy model for opponent inference.

    Mirrors the full training architecture:
        encoder (stem + pos_enc + res_blocks_1 + tile_attn_mid +
                 res_blocks_2 + tile_attn + enc_proj)
        → LSTM (temporal modeling across turns)
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

        raw_dim = conv_ch * NUM_TILES
        if enc_out_dim != raw_dim:
            self.enc_proj = nn.Sequential(
                nn.LayerNorm(raw_dim),
                nn.Linear(raw_dim, enc_out_dim),
            )
        else:
            self.enc_proj = None

        self.lstm = nn.LSTM(enc_out_dim, rnn_size, batch_first=True)
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
        x = self.res_blocks_1(x)
        x = self.tile_attn_mid(x)
        x = self.res_blocks_2(x)
        x = self.tile_attn(x)
        flat = x.reshape(B, -1)
        if self.enc_proj is not None:
            flat = self.enc_proj(flat)
        return flat

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
        logits[mask < 0.5] = -1e9
        if temperature <= 0.01:
            return int(logits.argmax().item()), new_hidden
        probs = F.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item()), new_hidden

    def init_hidden(self, device: str = "cpu") -> HiddenState:
        h = torch.zeros(1, 1, self._rnn_size, device=device)
        c = torch.zeros(1, 1, self._rnn_size, device=device)
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

        # Detect rnn_size from LSTM hidden-hidden weight shape
        lstm_hh_key = "core.rnn.weight_hh_l0"
        if lstm_hh_key in model_sd:
            rnn_size = model_sd[lstm_hh_key].shape[1]

        # Detect enc_out_dim from enc_proj if present
        enc_proj_key = "enc_proj.1.weight"
        if enc_proj_key in encoder_sd:
            enc_out_dim = encoder_sd[enc_proj_key].shape[0]
        else:
            enc_out_dim = conv_ch * NUM_TILES

        # Count blocks in each half (split architecture)
        num_blocks_1 = 0
        while f"res_blocks_1.{num_blocks_1}.block.0.weight" in encoder_sd:
            num_blocks_1 += 1
        num_blocks_2 = 0
        while f"res_blocks_2.{num_blocks_2}.block.0.weight" in encoder_sd:
            num_blocks_2 += 1
        num_blocks = num_blocks_1 + num_blocks_2
        if num_blocks == 0:
            log.warning("Could not detect num_blocks; defaulting to 20")
            num_blocks = 20

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
        )
        model.to(device)

        partial_sd = {}
        # Encoder weights
        for k, v in encoder_sd.items():
            if k in model.state_dict():
                partial_sd[k] = v

        # Map core.rnn.* → lstm.*
        for k, v in model_sd.items():
            if k.startswith("core.rnn."):
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
            "Loaded opponent model from %s (%d keys, rnn_size=%d, enc_out=%d)",
            path, len(partial_sd), rnn_size, enc_out_dim,
        )
        return model


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

    def load(self, path: str) -> None:
        path_str = str(path)
        if path_str == self._current_path:
            return
        try:
            self._model = PolicyModel.from_sf2_checkpoint(path_str, self._device)
            self._current_path = path_str
            self._hidden_states.clear()
            log.info("Loaded opponent model: %s", path_str)
        except Exception as e:
            log.warning(
                "Failed to load opponent model from %s: %s. "
                "Falling back to %s.",
                path, e,
                "previous model" if self._model is not None else "random policy",
            )

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
