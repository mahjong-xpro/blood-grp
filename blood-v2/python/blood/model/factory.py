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

from blood.consts import NUM_ORACLE_CHANNELS
from .encoder import SuitAwareResNetEncoder
from .heads import AuxHead
from .oracle import OracleEncoder, DistillationLoss
from .opponent_model import OpponentHandPredictor


class TurnAttention(nn.Module):
    """Turn-level cross-attention over LSTM history.

    Maintains a memory buffer of recent LSTM outputs and attends over them
    to recover information that may have been compressed in the LSTM hidden state.
    Uses residual connection so initial behavior is equivalent to pure LSTM.
    """

    def __init__(self, dim: int = 512, num_heads: int = 4, max_turns: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_turns, dim))
        self._max_turns = max_turns
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Initialize output projection near zero for smooth residual start
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        # Pre-register causal mask buffer (upper-triangular = masked positions)
        causal = torch.triu(torch.ones(max_turns, max_turns, dtype=torch.bool), diagonal=1)
        self.register_buffer("_causal_mask", causal, persistent=False)

    def forward(self, current: Tensor, memory: Tensor) -> Tensor:
        """
        Args:
            current: (B, 1, dim) — current turn LSTM output
            memory: (B, K, dim) — recent K turns of LSTM outputs
        Returns:
            (B, 1, dim) — attended output with residual
        """
        K = memory.size(1)
        memory = memory + self.pos_embed[:, :K]
        q = self.norm(current)
        k = v = self.norm(memory)
        attn_out, _ = self.attn(q, k, v)
        return current + attn_out

    def forward_causal(self, seq: Tensor) -> Tensor:
        """Causal self-attention over a full sequence in a single call.

        Args:
            seq: (B, T, dim) — full LSTM output sequence
        Returns:
            (B, T, dim) — attended output with residual
        """
        T = seq.size(1)
        seq_pos = seq + self.pos_embed[:, :T]
        q = self.norm(seq)
        k = v = self.norm(seq_pos)
        mask = self._causal_mask[:T, :T].to(device=seq.device)
        attn_out, _ = self.attn(q, k, v, attn_mask=mask)
        return seq + attn_out


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

        # DingQue progressive prior: linearly decay uniform prior over training
        self.dingque_prior_warmup_steps = getattr(cfg, "dingque_prior_warmup_steps", 100000)
        self.dingque_prior_enabled = getattr(cfg, "dingque_prior_enabled", True)
        # Global step counter for prior decay (updated externally by trainer)
        self.register_buffer("_dingque_global_steps", torch.tensor(0, dtype=torch.long), persistent=False)

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
        self.action_parameterization = self.get_action_parameterization(head_dim)
        self.critic_linear = nn.Linear(head_dim, 1)

        # AuxHead reads post-LSTM features (core_out) to avoid gradient conflict.
        # Pre-LSTM placement caused the encoder to be optimized for aux tasks rather
        # than LSTM temporal modeling. Post-LSTM placement incentivizes the LSTM to
        # maintain opponent state in its hidden state, which is the correct inductive bias.
        self.aux_head = AuxHead(
            in_dim=core_out, hidden=512,
            focal_alpha=getattr(cfg, "aux_focal_alpha", 0.25),
            focal_gamma=getattr(cfg, "aux_focal_gamma", 2.0),
        )
        self._aux_enabled = getattr(cfg, "aux_shanten_weight", 1.0) > 0
        self.shanten_weight = getattr(cfg, "aux_shanten_weight", 1.0)
        self.ow_weight = getattr(cfg, "aux_opp_waits_weight", 0.3)  # Fix R11-L1: match cfg.py default

        self.oracle_enabled = getattr(cfg, "oracle_enabled", True)
        if self.oracle_enabled:
            oracle_obs_ch = NUM_ORACLE_CHANNELS
            oracle_blocks = getattr(cfg, "oracle_num_blocks", 20)
            oracle_attn_layers = getattr(cfg, "oracle_num_tile_attn_layers", 4)
            oracle_attn_heads = getattr(cfg, "oracle_tile_attn_heads", 4)
            self.oracle_encoder = OracleEncoder(
                obs_channels=oracle_obs_ch,
                conv_ch=256,
                num_blocks=oracle_blocks,
                action_dim=action_space.n,
                num_tile_attn_layers=oracle_attn_layers,
                tile_attn_heads=oracle_attn_heads,
            )
            self.distill_loss_fn = DistillationLoss(
                temperature=getattr(cfg, "oracle_distill_temperature", 2.0),
            )
            self.distill_weight = getattr(cfg, "oracle_distill_weight", 0.05)
            self.oracle_ce_weight = getattr(cfg, "oracle_ce_weight", 0.1)
            self.oracle_value_distill_weight = getattr(cfg, "oracle_value_distill_weight", 0.0)
            self.oracle_value_warmup_steps = getattr(cfg, "oracle_value_warmup_steps", 500_000)
            self.oracle_value_head_loss_weight = getattr(cfg, "oracle_value_head_loss_weight", 1.0)

        # Opponent hand predictor (A3): lightweight model trained with Oracle labels
        self.opponent_predictor_enabled = getattr(cfg, "opponent_predictor_enabled", False)
        if self.opponent_predictor_enabled:
            opp_conv_ch = getattr(cfg, "opponent_predictor_conv_ch", 128)
            opp_blocks = getattr(cfg, "opponent_predictor_num_blocks", 6)
            self.opponent_predictor = OpponentHandPredictor(
                conv_ch=opp_conv_ch,
                num_blocks=opp_blocks,
            )
            self.opponent_predictor_weight = getattr(cfg, "opponent_predictor_weight", 0.1)

        # TurnAttention (B1): cross-attention over LSTM history
        self.turn_attn_enabled = getattr(cfg, "turn_attention_enabled", False)
        if self.turn_attn_enabled:
            ta_heads = getattr(cfg, "turn_attention_heads", 4)
            ta_max_turns = getattr(cfg, "recurrence", 32)
            self.turn_attention = TurnAttention(
                dim=core_out, num_heads=ta_heads, max_turns=ta_max_turns,
            )

        self._cached_encoder_out = None  # post-enc_proj; used as forward-pass guard in runner.py
        self._cached_core_out = None     # post-LSTM; used by AuxHead in runner.py
        self._cached_values = None       # student values; used by Oracle value distillation
        self._cached_obs = None
        self._cache_gen = 0
        self._loss_gen = -1  # -1 so first batch (cache_gen=1 > loss_gen=-1) always passes

        # Fix R11-M2: re-apply orthogonal init to heads created after super().__init__().
        # SF2's self.apply(initialize_weights) runs inside super().__init__() before
        # actor_head/critic_head/aux_head exist, so they get PyTorch default (Kaiming).
        self.apply(self.initialize_weights)

        # Fix R12-H4: re-apply TurnAttention zero-init after orthogonal init.
        # TurnAttention.out_proj is designed to start near zero for smooth residual,
        # but self.apply(initialize_weights) overwrites it with orthogonal init.
        if self.turn_attn_enabled and hasattr(self, "turn_attention"):
            nn.init.zeros_(self.turn_attention.attn.out_proj.weight)
            nn.init.zeros_(self.turn_attention.attn.out_proj.bias)

    def reset_cache_counters(self):
        """Reset cache generation counters after checkpoint reload (Issue #39)."""
        self._cache_gen = 0
        self._loss_gen = -1

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        # trunk features
        x = self.encoder(normalized_obs_dict)
        self._cached_encoder_out = x
        self._cached_obs = normalized_obs_dict
        self._cache_gen += 1
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        # During BPTT training, SF2 passes head_output as a PackedSequence and
        # expects a PackedSequence back. Do NOT apply heads here — they require
        # a plain Tensor. SF2 unpacks the output before calling forward_tail.
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        # core_output is always a plain Tensor here (SF2 unpacks before calling us).

        # B1: TurnAttention — apply cross-attention over LSTM sequence
        # During BPTT training, core_output is (B*T, dim); during inference, (B, dim).
        # Training uses forward_causal for O(T) attention calls instead of O(T²) loop.
        if getattr(self, "turn_attn_enabled", False) and hasattr(self, "turn_attention"):
            recurrence = getattr(self.cfg, "recurrence", 32) if hasattr(self, "cfg") else 32
            total = core_output.shape[0]
            dim = core_output.shape[-1]
            if total > 1 and total % recurrence == 0:
                # Training mode: single causal attention call over full sequence
                B = total // recurrence
                T = recurrence
                seq = core_output.view(B, T, dim)
                core_output = self.turn_attention.forward_causal(seq).reshape(total, dim)
            elif total == 1:
                # Single-sample inference: self-attend (identity due to zero-init)
                core_output = self.turn_attention(
                    core_output.unsqueeze(1), core_output.unsqueeze(1)
                ).squeeze(1)
            # else: non-aligned batch — skip TurnAttention to avoid cross-contamination

        self._cached_core_out = core_output  # post-TurnAttention features for AuxHead
        actor_features = self.actor_head(core_output)
        critic_features = self.critic_head(core_output)

        values = self.critic_linear(critic_features).squeeze(-1)
        self._cached_values = values
        result = TensorDict(values=values)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_features)

        # DingQue progressive prior: mix model output with uniform prior
        if self.dingque_prior_enabled and self._cached_obs is not None:
            mask = self._cached_obs.get("action_mask")
            if mask is not None:
                # Detect DingQue phase: actions 31-33 are valid
                dingque_mask = (mask[:, 31:34].sum(dim=1) > 0.5)
                
                if dingque_mask.any():
                    # Compute prior strength: linearly decay from 1.0 to 0.0
                    global_steps = float(self._dingque_global_steps.item())
                    prior_strength = max(0.0, 1.0 - global_steps / self.dingque_prior_warmup_steps)
                    
                    if prior_strength > 0.0:
                        # Uniform prior logits for DingQue actions (31-33)
                        uniform_logits = torch.zeros(3, dtype=action_distribution_params.dtype, device=action_distribution_params.device)
                        
                        # Mix model output with uniform prior
                        # logits_mixed = (1 - α) * logits_model + α * logits_uniform
                        action_distribution_params[dingque_mask, 31:34] = (
                            (1.0 - prior_strength) * action_distribution_params[dingque_mask, 31:34] +
                            prior_strength * uniform_logits
                        )

        # 对非法动作施加掩码，使其永远不会被采样，且在 PPO loss 中贡献近零概率。
        # 使用 dtype.min 替代硬编码 -1e9，避免 float16 下溢出为 -inf
        # （float16 最大值约 65504，-1e9 会溢出），确保混合精度训练数值稳定。
        # _cached_obs 在 forward_head() 中为当前 minibatch 设置。
        if self._cached_obs is not None:
            mask = self._cached_obs.get("action_mask")
            if mask is not None:
                illegal = ~mask.bool()
                mask_value = torch.finfo(action_distribution_params.dtype).min
                action_distribution_params = action_distribution_params.masked_fill(illegal, mask_value)
                
                # Sync raw_logits so SF2's entropy/log_prob use the masked distribution.
                self.last_action_distribution.raw_logits = action_distribution_params

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
