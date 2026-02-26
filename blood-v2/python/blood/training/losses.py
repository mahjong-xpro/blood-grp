"""Blood auxiliary loss computation, extracted from runner._patch_learner().

BloodLossComputer encapsulates all custom losses injected into the SF2 training
loop so that runner.py stays a thin monkey-patch wrapper.
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor

from blood.consts import (
    NUM_STUDENT_CHANNELS, NUM_TILE_TYPES,
    CH_OPP_KAWA_BASE, CH_OPP_KAWA_STRIDE,
    CH_VISIBLE_TILES_BASE, CH_WALL_REMAINING, CH_TURN_PROGRESS,
    CH_OPP_DING_QUE_BASE, CH_OPP_MELD_BASE, CH_OPP_HAND_INFO_BASE,
    CH_GENBUTSU_BASE,
)

log = logging.getLogger(__name__)


class BloodLossComputer:
    """Computes all Blood-specific auxiliary losses for a single PPO minibatch.

    Called from the monkey-patched Learner._calculate_losses after SF2 has
    computed its own PPO losses.  No state is stored between calls — all
    inputs come from the actor-critic's cached forward-pass tensors.
    """

    def __init__(self, cfg=None):
        self._metrics_interval = getattr(cfg, 'blood_metrics_interval', 100) if cfg else 100
        self._metrics_step = 0
        self._cached_logprob_metrics: dict = {}

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
        extra_loss = torch.zeros((), device=device)

        if features is None or obs is None or not ac.training:
            return extra_loss, summaries

        # Stale-cache guard: skip if the encoder cache wasn't refreshed this step.
        # Use != instead of <= to handle checkpoint reload where counters may
        # reset independently (Issue #39).
        cache_gen = getattr(ac, "_cache_gen", 0)
        loss_gen = getattr(ac, "_loss_gen", -1)
        if cache_gen == loss_gen:
            log.warning(
                "Stale encoder cache detected (gen %d == %d); skipping aux losses",
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

        # --- Opponent hand prediction loss (A3) ---
        if getattr(ac, "opponent_predictor_enabled", False):
            opp_loss = self._opponent_hand_loss(ac, obs)
            if opp_loss is not None:
                opp_weight = getattr(ac, "opponent_predictor_weight", 0.1)
                extra_loss = extra_loss + opp_weight * opp_loss
                summaries["opponent_pred_loss"] = opp_loss.detach()

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
        oracle_logits_masked = oracle_logits
        if action_mask is not None:
            # Use dtype.min for consistent masking (same strategy as factory.py forward_tail)
            mask_value = torch.finfo(oracle_logits.dtype).min
            oracle_logits_masked = oracle_logits.masked_fill(
                ~action_mask.bool(), mask_value
            )
        ce_raw = F.cross_entropy(oracle_logits_masked, actions.long(), reduction="none")
        if advantages is not None:
            # Use softmax weighting instead of clamp(min=0) to ensure non-zero
            # weights even when all advantages are negative (Issue #45).
            # Floor adv_std at 0.1 (not 1e-4) to prevent overflow in float16
            # where exp(x) overflows for x > ~11. Clamp normalized advantages
            # to [-10, 10] as an additional safety net.
            adv_std = max(advantages.detach().std().item(), 0.1)
            normed = (advantages.detach() / adv_std).clamp(-10.0, 10.0)
            adv_w = F.softmax(normed, dim=0)
            adv_w = adv_w * len(adv_w)  # Scale so mean weight ≈ 1.0
            return (ce_raw * adv_w).mean()
        return ce_raw.mean()

    def _oracle_value_head_loss(self, oracle_values, returns) -> Tensor | None:
        """MSE between oracle value head predictions and GAE returns."""
        if returns is None:
            return None
        ov = oracle_values.squeeze(-1).view(-1)
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
        ov = oracle_values.squeeze(-1).detach().view(-1)
        if sv.shape != ov.shape:
            log.warning(
                "oracle_value_distill_loss skipped: shape mismatch student=%s oracle=%s",
                sv.shape, ov.shape,
            )
            return None
        return F.mse_loss(sv, ov)

    def _opponent_hand_loss(self, ac, obs) -> Tensor | None:
        """BCE loss for opponent hand prediction using Oracle obs as labels.

        Extracts opponent hand ground truth from Oracle observation channels
        (first 12 channels of oracle_extra = 3 opponents × 4 one-hot hand counts).
        Constructs input features from student obs (opponent kawa + melds + visible + context).
        """
        oracle_obs = obs.get("oracle_obs")
        if oracle_obs is None:
            return None

        predictor = getattr(ac, "opponent_predictor", None)
        if predictor is None:
            return None

        student_obs = obs.get("obs")
        if student_obs is None:
            return None

        # Extract opponent hand ground truth from oracle extra channels.
        # Oracle obs shape: (B, oracle_channels * 27)
        # The first 12 extra channels (after student channels) encode opponent hands
        # as 4 one-hot layers per opponent (3 opponents × 4 = 12 channels).
        B = oracle_obs.shape[0]
        oracle_2d = oracle_obs.view(B, -1, NUM_TILE_TYPES)  # (B, oracle_ch, 27)
        student_2d = student_obs.view(B, -1, NUM_TILE_TYPES)
        student_ch = NUM_STUDENT_CHANNELS

        # Section 7 fuuro layout: kawa_overview(3×4=12) + fuuro(4×8=32) + ankan(1) + ...
        # Fuuro starts at CH_VISIBLE_TILES_BASE + 12; self=8ch then 3 opponents×8ch each
        fuuro_base = CH_VISIBLE_TILES_BASE + 12  # skip 3×4 kawa overview
        opp_fuuro_base = fuuro_base + 8  # skip self fuuro (8ch)

        # Ground truth: sum of 4 one-hot channels per opponent → tile count [0,4]
        # Normalize to [0,1] by dividing by 4
        # Pre-allocate feature buffer once, reuse per opponent to reduce allocation
        opp_features = torch.zeros(B, 75, NUM_TILE_TYPES, device=oracle_obs.device)
        total_loss = torch.tensor(0.0, device=oracle_obs.device)
        for opp_idx in range(3):
            opp_hand_gt = oracle_2d[:, student_ch + opp_idx * 4 : student_ch + (opp_idx + 1) * 4, :].sum(dim=1)
            opp_hand_gt = (opp_hand_gt / 4.0).clamp(0, 1)  # (B, 27)

            # Build 75ch input: kawa(58) + visible(4) + fuuro(8) + context(5)
            opp_features.zero_()  # reuse buffer
            ch = 0

            # [0:58] Opponent kawa (58 ch) from Section 6
            kawa_start = CH_OPP_KAWA_BASE + opp_idx * CH_OPP_KAWA_STRIDE
            kawa_end = kawa_start + CH_OPP_KAWA_STRIDE
            if kawa_end <= student_2d.shape[1]:
                opp_features[:, ch:ch + CH_OPP_KAWA_STRIDE, :] = student_2d[:, kawa_start:kawa_end, :]
            ch += CH_OPP_KAWA_STRIDE  # 58

            # [58:62] Visible tiles / kawa overview (4 ch) from Section 7
            vis_start = CH_VISIBLE_TILES_BASE + opp_idx * 4
            vis_end = vis_start + 4
            if vis_end <= student_2d.shape[1]:
                opp_features[:, ch:ch + 4, :] = student_2d[:, vis_start:vis_end, :]
            ch += 4  # 62

            # [62:70] Opponent fuuro (8 ch) from Section 7: 4 melds × 2ch (tile + type)
            meld_start = opp_fuuro_base + opp_idx * 8
            meld_end = meld_start + 8
            if meld_end <= student_2d.shape[1]:
                opp_features[:, ch:ch + 8, :] = student_2d[:, meld_start:meld_end, :]
            ch += 8  # 70

            # [70] Wall remaining (1 ch) from Section 4
            if CH_WALL_REMAINING < student_2d.shape[1]:
                opp_features[:, ch, :] = student_2d[:, CH_WALL_REMAINING, :]
            ch += 1  # 71

            # [71] Turn progress (1 ch) from Section 2
            if CH_TURN_PROGRESS < student_2d.shape[1]:
                opp_features[:, ch, :] = student_2d[:, CH_TURN_PROGRESS, :]
            ch += 1  # 72

            # [72:75] Opponent ding-que (3 ch) from Section 3
            dq_start = CH_OPP_DING_QUE_BASE + opp_idx * 3
            dq_end = dq_start + 3
            if dq_end <= student_2d.shape[1]:
                opp_features[:, ch:ch + 3, :] = student_2d[:, dq_start:dq_end, :]
            ch += 3  # 75

            pred = predictor(opp_features)
            total_loss = total_loss + predictor.loss(pred, opp_hand_gt)

        return total_loss / 3.0  # average over 3 opponents

    def _logprob_metrics(self, action_dist, obs, summaries) -> None:
        """Monitor logprob extremes over legal actions only.

        The log_softmax computation is throttled to every ``_metrics_interval``
        steps to avoid unnecessary overhead on every minibatch.  Between
        computation steps the last cached values are returned.
        """
        self._metrics_step += 1
        if self._metrics_step % self._metrics_interval != 0:
            # Return cached metrics from the last computation
            summaries.update(self._cached_logprob_metrics)
            return

        student_logits = getattr(action_dist, "raw_logits", None)
        if student_logits is None:
            return
        mask = obs.get("action_mask") if obs is not None else None
        with torch.no_grad():
            log_probs = F.log_softmax(student_logits.detach(), dim=-1)
            if mask is not None:
                legal_lp = log_probs[mask.bool()]
                metrics = {
                    "blood/max_abs_logprob": legal_lp.abs().max(),
                    "blood/mean_abs_logprob": legal_lp.abs().mean(),
                }
            else:
                metrics = {
                    "blood/max_abs_logprob": log_probs.abs().max(),
                    "blood/mean_abs_logprob": log_probs.abs().mean(),
                }
        self._cached_logprob_metrics = metrics
        summaries.update(metrics)
