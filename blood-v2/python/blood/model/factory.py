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

        enc_out = self.encoder.get_out_size()
        head_dim = 512
        
        # NOTE: Decoupled heads for Actor and Critic
        self.actor_head = nn.Sequential(
            nn.Linear(enc_out, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(enc_out, head_dim),
            nn.Mish(inplace=True),
            nn.LayerNorm(head_dim),
        )

        # Replace standard SF heads with our decoupled ones
        # SF2 ActorCriticSharedWeights uses:
        # self.action_parameterization = ... (built in super().__init__)
        # self.critic_linear = nn.Linear(..., 1)
        
        # We need to re-bind the input dimensions for these final layers
        self.action_parameterization = model_factory.make_action_parameterization(cfg, head_dim, action_space)
        self.critic_linear = nn.Linear(head_dim, 1)

        self.aux_head = AuxHead(in_dim=enc_out, hidden=512)
        self._aux_enabled = getattr(cfg, "aux_dingque_weight", 1.0) > 0
        self.dq_weight = getattr(cfg, "aux_dingque_weight", 1.0)
        self.ow_weight = getattr(cfg, "aux_opp_waits_weight", 0.1)

        self.oracle_enabled = getattr(cfg, "oracle_enabled", True)
        if self.oracle_enabled:
            oracle_obs_ch = 430
            oracle_blocks = getattr(cfg, "oracle_num_blocks", 20)  # Also 20 blocks for oracle
            self.oracle_encoder = OracleEncoder(
                obs_channels=oracle_obs_ch,
                conv_ch=256,         # Also 256 for oracle
                num_blocks=oracle_blocks,
                out_dim=enc_out,     # Match student enc_out
                action_dim=action_space.n,
            )
            self.distill_loss_fn = DistillationLoss(
                temperature=getattr(cfg, "oracle_distill_temperature", 2.0),
            )
            self.distill_weight = getattr(cfg, "oracle_distill_weight", 0.05)
            self.oracle_ce_weight = getattr(cfg, "oracle_ce_weight", 0.1)

        self._cached_encoder_out = None
        self._cached_obs = None
        self._cache_gen = 0
        self._loss_gen = 0

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        # trunk features
        x = self.encoder(normalized_obs_dict)
        self._cached_encoder_out = x
        self._cached_obs = normalized_obs_dict
        self._cache_gen += 1
        return x

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        
        # Decoupled pass
        if values_only:
            v_head = self.critic_head(x)
            v = self.critic_linear(v_head)
            return TensorDict(values=v)
        
        a_head = self.actor_head(x)
        action_dist = self.action_parameterization(a_head)
        
        v_head = self.critic_head(x)
        v = self.critic_linear(v_head)
        
        return TensorDict(
            action_logits=action_dist.logits, # Sample Factory and our DistillCallback need this
            action_dist=action_dist,
            values=v,
        )


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
