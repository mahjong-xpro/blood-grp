"""Blood auxiliary loss computation, extracted from runner._patch_learner().

BloodLossComputer encapsulates all custom losses injected into the SF2 training
loop so that runner.py stays a thin monkey-patch wrapper.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


class BloodLossComputer:
    """Computes all Blood-specific auxiliary losses for a single PPO minibatch.

    Called from the monkey-patched Learner._calculate_losses after SF2 has
    computed its own PPO losses.  No state is stored between calls — all
    inputs come from the actor-critic's cached forward-pass tensors.
    """

    def compute(
        self,
        ac,
        mb,
        action_dist,
        value_loss: Tensor,
        summaries: dict,
        env_steps: int = 0,
    ) -> tuple[Tensor, dict]:
        """Return (extra_loss, summaries) for the current minibatch.

        extra_loss is a scalar tensor on the same device as value_loss.
        summaries is mutated in-place and also returned for convenience.
        """
        features = getattr(ac, "_cached_encoder_out", None)
        core_features = getattr(ac, "_cached_core_out", None)
        obs = getattr(ac, "_cached_obs", None)

        device = value_loss.device if hasattr(value_loss, "device") else "cpu"
        extra_loss = torch.zeros(1, device=device)

        if features is None or obs is None or not ac.training:
            return extra_loss, summaries

        # Stale-cache guard: skip if the encoder cache wasn't refreshed this step.
        cache_gen = getattr(ac, "_cache_gen", 0)
        loss_gen = getattr(ac, "_loss_gen", 0)
        if cache_gen <= loss_gen:
            log.warning(
                "Stale encoder cache detected (gen %d <= %d); skipping aux losses",
                cache_gen, loss_gen,
            )
            return extra_loss, summaries
        ac._loss_gen = cache_gen

        # --- Aux head (shanten + ow) ---
        if getattr(ac, "_aux_enabled", False):
            aux = self._aux_loss(ac, core_features, obs)
            if aux is not None:
                extra_loss = extra_loss + aux
                summaries["aux_loss"] = aux.detach()

        # --- Oracle distillation block ---
        if getattr(ac, "oracle_enabled", False):
            oracle_obs = obs.get("oracle_obs")
            action_mask = obs.get("action_mask")
            if oracle_obs is not None:
                oracle_logits, oracle_values = ac.oracle_encoder(oracle_obs)

                # KL distillation: student → oracle
                student_logits = getattr(action_dist, "raw_logits", None)
                if student_logits is None:
                    log.warning(
                        "Cannot find 'raw_logits' on action_dist; skipping distillation"
                    )
                else:
                    mask_bool = action_mask.bool() if action_mask is not None else None
                    distill = self._oracle_distill_loss(
                        ac, student_logits, oracle_logits, mask_bool
                    )
                    extra_loss = extra_loss + ac.distill_weight * distill
                    summaries["distill_loss"] = distill.detach()

                # Oracle CE (advantage-weighted)
                oracle_ce = self._oracle_ce_loss(
                    oracle_logits, mb.actions, getattr(mb, "advantages", None), action_mask
                )
                oracle_ce_weight = getattr(ac, "oracle_ce_weight", 0.1)
                extra_loss = extra_loss + oracle_ce_weight * oracle_ce
                summaries["oracle_ce"] = oracle_ce.detach()

                # Oracle value head supervised loss
                oracle_value_head_weight = getattr(ac, "oracle_value_head_loss_weight", 1.0)
                if oracle_value_head_weight > 0:
                    ovh = self._oracle_value_head_loss(oracle_values, getattr(mb, "returns", None))
                    if ovh is not None:
                        extra_loss = extra_loss + oracle_value_head_weight * ovh
                        summaries["oracle_value_head_loss"] = ovh.detach()

                # Oracle value distillation (gated by warmup)
                ovd = self._oracle_value_distill_loss(ac, oracle_values, env_steps)
                if ovd is not None:
                    oracle_value_distill_weight = getattr(ac, "oracle_value_distill_weight", 0.0)
                    extra_loss = extra_loss + oracle_value_distill_weight * ovd
                    summaries["oracle_value_distill_loss"] = ovd.detach()

        # --- Logprob metrics (legal actions only) ---
        self._logprob_metrics(action_dist, obs, summaries)

        return extra_loss, summaries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aux_loss(self, ac, core_features, obs) -> Tensor | None:
        shanten_labels = obs.get("shanten_labels")
        ow_labels = obs.get("ow_labels")
        if shanten_labels is None or ow_labels is None or core_features is None:
            return None
        return ac.aux_head.loss(
            core_features, shanten_labels, ow_labels,
            shanten_weight=ac.shanten_weight, ow_weight=ac.ow_weight,
        )

    def _oracle_distill_loss(self, ac, student_logits, oracle_logits, mask_bool) -> Tensor:
        return ac.distill_loss_fn(student_logits, oracle_logits.detach(), mask_bool)

    def _oracle_ce_loss(self, oracle_logits, actions, advantages, action_mask) -> Tensor:
        """Advantage-weighted cross-entropy on oracle logits vs taken actions."""
        oracle_logits_masked = oracle_logits.clone()
        if action_mask is not None:
            oracle_logits_masked = oracle_logits_masked.masked_fill(
                ~action_mask.bool(), torch.finfo(oracle_logits.dtype).min
            )
        ce_raw = F.cross_entropy(oracle_logits_masked, actions.long(), reduction="none")
        if advantages is not None:
            adv_w = torch.clamp(advantages.detach(), min=0.0)
            adv_w = adv_w / (adv_w.mean() + 1e-8)
            return (ce_raw * adv_w).mean()
        return ce_raw.mean()

    def _oracle_value_head_loss(self, oracle_values, returns) -> Tensor | None:
        """MSE between oracle value head predictions and GAE returns."""
        if returns is None:
            return None
        ov = oracle_values.squeeze().view(-1)
        ret = returns.view(-1)
        if ov.shape != ret.shape:
            log.warning(
                "oracle_value_head_loss skipped: shape mismatch oracle_values=%s returns=%s",
                ov.shape, ret.shape,
            )
            return None
        return F.mse_loss(ov, ret.detach())

    def _oracle_value_distill_loss(self, ac, oracle_values, env_steps) -> Tensor | None:
        """MSE between student critic and oracle value head (gated by warmup)."""
        oracle_value_distill_weight = getattr(ac, "oracle_value_distill_weight", 0.0)
        oracle_value_warmup = getattr(ac, "oracle_value_warmup_steps", 500_000)
        if oracle_value_distill_weight <= 0 or env_steps < oracle_value_warmup:
            return None
        student_values = getattr(ac, "_cached_values", None)
        if student_values is None:
            return None
        sv = student_values.view(-1)
        ov = oracle_values.squeeze().detach().view(-1)
        if sv.shape != ov.shape:
            log.warning(
                "oracle_value_distill_loss skipped: shape mismatch student=%s oracle=%s",
                sv.shape, ov.shape,
            )
            return None
        return F.mse_loss(sv, ov)

    def _logprob_metrics(self, action_dist, obs, summaries) -> None:
        """Monitor logprob extremes over legal actions only."""
        student_logits = getattr(action_dist, "raw_logits", None)
        if student_logits is None:
            return
        mask = obs.get("action_mask") if obs is not None else None
        with torch.no_grad():
            log_probs = F.log_softmax(student_logits, dim=-1)
            if mask is not None:
                legal_lp = log_probs[mask.bool()]
                summaries["blood/max_abs_logprob"] = legal_lp.abs().max()
                summaries["blood/mean_abs_logprob"] = legal_lp.abs().mean()
            else:
                summaries["blood/max_abs_logprob"] = log_probs.abs().max()
                summaries["blood/mean_abs_logprob"] = log_probs.abs().mean()
