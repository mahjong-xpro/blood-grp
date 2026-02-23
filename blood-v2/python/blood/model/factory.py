"""Sample Factory model and training registration."""

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic, ActorCriticSharedWeights
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace

from .encoder import SuitAwareResNetEncoder
from .heads import AuxHead
from .oracle import OracleEncoder, DistillationLoss


class BloodActorCritic(ActorCriticSharedWeights):
    """Custom ActorCritic with auxiliary heads and oracle distillation.

    - AuxHead: predicts opponent dingque + waiting tiles
    - OracleEncoder: perfect-info teacher for policy distillation
    Both losses are computed in the monkey-patched _calculate_losses.
    """

    def __init__(self, model_factory, obs_space, action_space, cfg):
        super().__init__(model_factory, obs_space, action_space, cfg)

        core_out = self.core.get_out_size()     # 1024 with LSTM, 6912 with Identity core
        head_dim = 512

        # Pre-norm 2-layer heads: LayerNorm before each Linear for training stability.
        # Pre-norm (LN → Linear → Mish) is more stable than Post-norm (Linear → Mish → LN)
        # because gradients are normalized before entering each linear layer.
        self.actor_head = nn.Sequential(
            nn.LayerNorm(core_out),
            nn.Linear(core_out, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(inplace=True),
        )
        self.critic_head = nn.Sequential(
            nn.LayerNorm(core_out),
            nn.Linear(core_out, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, head_dim),
            nn.Mish(inplace=True),
        )

        # Replace standard SF heads with our decoupled ones
        # SF2 ActorCriticSharedWeights uses:
        # self.action_parameterization = ... (built in super().__init__)
        # self.critic_linear = nn.Linear(..., 1)
        
        # We need to re-bind the input dimensions for these final layers
        self.action_parameterization = model_factory.make_action_parameterization(cfg, head_dim, action_space)
        self.critic_linear = nn.Linear(head_dim, 1)

        # AuxHead reads post-LSTM features (core_out) to avoid gradient conflict.
        # Pre-LSTM placement caused the encoder to be optimized for aux tasks rather
        # than LSTM temporal modeling. Post-LSTM placement incentivizes the LSTM to
        # maintain opponent state in its hidden state, which is the correct inductive bias.
        self.aux_head = AuxHead(in_dim=core_out, hidden=512)
        self._aux_enabled = getattr(cfg, "aux_shanten_weight", 1.0) > 0
        self.shanten_weight = getattr(cfg, "aux_shanten_weight", 1.0)
        self.ow_weight = getattr(cfg, "aux_opp_waits_weight", 0.1)

        self.oracle_enabled = getattr(cfg, "oracle_enabled", True)
        if self.oracle_enabled:
            oracle_obs_ch = 516
            oracle_blocks = getattr(cfg, "oracle_num_blocks", 25)  # 25 blocks for oracle (matches cfg.py default)
            self.oracle_encoder = OracleEncoder(
                obs_channels=oracle_obs_ch,
                conv_ch=256,         # Also 256 for oracle
                num_blocks=oracle_blocks,
                action_dim=action_space.n,
            )
            self.distill_loss_fn = DistillationLoss(
                temperature=getattr(cfg, "oracle_distill_temperature", 2.0),
            )
            self.distill_weight = getattr(cfg, "oracle_distill_weight", 0.05)
            self.oracle_ce_weight = getattr(cfg, "oracle_ce_weight", 0.1)
            self.oracle_value_distill_weight = getattr(cfg, "oracle_value_distill_weight", 0.5)

        self._cached_encoder_out = None  # post-enc_proj (1024); used as forward-pass guard in runner.py
        self._cached_core_out = None     # post-LSTM (1024); used by AuxHead
        self._cached_values = None       # student values; used by Oracle value distillation
        self._cached_obs = None
        self._actor_features = None
        self._critic_features = None
        self._cache_gen = 0
        self._loss_gen = -1  # -1 so first batch (cache_gen=1 > loss_gen=-1) always passes

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        # trunk features
        x = self.encoder(normalized_obs_dict)
        self._cached_encoder_out = x
        self._cached_obs = normalized_obs_dict
        self._cache_gen += 1
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        # Pass through base RNN core, then split into actor/critic branches.
        x, new_rnn_states = self.core(head_output, rnn_states)
        self._cached_core_out = x  # post-LSTM features for AuxHead
        self._actor_features = self.actor_head(x)
        self._critic_features = self.critic_head(x)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        # Use decoupled heads stored by forward_core.
        # Fall back to core_output if forward_core was not called (e.g. during export).
        a_feat = getattr(self, "_actor_features", None)
        c_feat = getattr(self, "_critic_features", None)
        if a_feat is None or c_feat is None:
            a_feat = self.actor_head(core_output)
            c_feat = self.critic_head(core_output)

        values = self.critic_linear(c_feat).squeeze()
        self._cached_values = values  # cache for Oracle value distillation in runner.py
        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(a_feat)
        result["action_logits"] = action_distribution_params
        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True)
        result["new_rnn_states"] = new_rnn_states
        return result


def make_blood_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    return SuitAwareResNetEncoder(cfg, obs_space)


def make_blood_actor_critic(
    cfg: Config, obs_space: ObsSpace, action_space: ActionSpace,
) -> ActorCritic:
    model_factory = global_model_factory()
    return BloodActorCritic(model_factory, obs_space, action_space, cfg)


def register_blood_model():
    factory = global_model_factory()
    factory.register_encoder_factory(make_blood_encoder)
    factory.register_actor_critic_factory(make_blood_actor_critic)
