"""End-to-end smoke tests for Blood Mahjong training pipeline.

Validates the full pipeline: model creation → environment interaction →
forward pass → loss computation → gradient update.

These tests use minimal model configs (64 channels, 2 res blocks) for fast
CPU execution. Tests that require the compiled Rust engine are skipped
automatically when it is unavailable.
"""

import sys
import math

import pytest
import torch
import torch.nn as nn
import numpy as np

from blood.model.encoder import (
    SuitAwareResNetEncoder,
    NUM_TILES,
    DEFAULT_OBS_CHANNELS,
)
from blood.model.heads import AuxHead
from blood.model.inference import PolicyModel
from blood.consts import (
    NUM_TILE_TYPES, ACTION_SPACE, NUM_STUDENT_CHANNELS,
    NUM_ORACLE_CHANNELS, OBS_SIZE, ORACLE_OBS_SIZE,
)

# ── Smoke test constants ─────────────────────────────────────────────────────
SMOKE_OBS_CHANNELS = 470
SMOKE_CONV_CH = 64
SMOKE_NUM_BLOCKS = 2
SMOKE_ENC_OUT = 128
SMOKE_BATCH = 4

# ── Check Rust engine availability ───────────────────────────────────────────
_rust_available = False
try:
    from blood._engine import RustMahjongEnv
    _rust_available = True
except ImportError:
    pass

requires_rust = pytest.mark.skipif(
    not _rust_available,
    reason="Rust engine (blood._engine) not compiled; skipping env tests",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_smoke_cfg(use_rnn=False):
    """Build a minimal cfg namespace matching smoke_test.yaml parameters.

    Mirrors the pattern in TestBloodActorCriticDims._make_cfg but with
    smoke-test-sized parameters. Does NOT depend on Sample Factory's
    argument parser, so it works without SF2 installed for basic tests.
    """
    class SmokeCfg:
        # Model
        blood_obs_channels = SMOKE_OBS_CHANNELS
        blood_conv_channels = SMOKE_CONV_CH
        blood_num_res_blocks = SMOKE_NUM_BLOCKS
        blood_encoder_out_dim = SMOKE_ENC_OUT
        blood_enc_proj_layers = 1
        blood_num_tile_attn_layers = 1
        blood_tile_attn_heads = 4

        # Oracle — disabled
        oracle_enabled = False

        # Env
        initial_score = 100_000
        opponent_mode = "rulebot"
        suit_augment_prob = 0.0

        # Reward shaping — all off
        warmup_reward_shaping = False
        reward_tsumo_bonus = 0.0
        reward_deal_in_penalty = 0.0
        reward_shanten_progress = 0.0
        reward_shanten_regress = 0.0
        shanten_fan_bonus_scale = 0.0
        reward_rank_bonus = 0.0
        reward_safe_discard = 0.0

        # Aux
        aux_shanten_weight = 0.1
        aux_opp_waits_weight = 0.1

    return SmokeCfg()


def _make_smoke_encoder():
    """Create a minimal SuitAwareResNetEncoder for smoke tests."""
    cfg = _make_smoke_cfg()
    return SuitAwareResNetEncoder(cfg, obs_space=None)


def _random_obs(batch=SMOKE_BATCH):
    """Generate random observation tensor shaped like the environment output."""
    return torch.randn(batch, SMOKE_OBS_CHANNELS * NUM_TILES)


def _random_action_mask(batch=SMOKE_BATCH):
    """Generate a random but valid action mask (at least one action legal)."""
    mask = torch.zeros(batch, ACTION_SPACE)
    for i in range(batch):
        # Ensure at least one legal action per sample
        n_legal = torch.randint(1, 5, (1,)).item()
        indices = torch.randperm(ACTION_SPACE)[:n_legal]
        mask[i, indices] = 1.0
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Test 1: Model Creation
# ══════════════════════════════════════════════════════════════════════════════

class TestModelCreation:
    """Verify model can be instantiated with smoke-test config and has
    a reasonable number of parameters."""

    def test_encoder_creation(self):
        """SuitAwareResNetEncoder should instantiate with minimal config."""
        enc = _make_smoke_encoder()
        assert enc.get_out_size() == SMOKE_ENC_OUT

    def test_encoder_param_count(self):
        """Smoke-sized encoder should have far fewer params than production."""
        enc = _make_smoke_encoder()
        n_params = sum(p.numel() for p in enc.parameters())
        # 64ch/2blocks should be well under 1M params
        assert 1_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"

    def test_policy_model_creation(self):
        """PolicyModel (inference wrapper) should instantiate with smoke params."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        assert model is not None
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_aux_head_creation(self):
        """AuxHead should instantiate with encoder output dim."""
        head = AuxHead(in_dim=SMOKE_ENC_OUT, hidden=64)
        assert head is not None


# ══════════════════════════════════════════════════════════════════════════════
# Test 2: Environment Creation (requires Rust engine)
# ══════════════════════════════════════════════════════════════════════════════

class TestEnvCreation:
    """Verify environment can be created, reset, and stepped."""

    @requires_rust
    def test_env_reset(self):
        """BloodMahjongEnv should reset and return valid observation dict."""
        from blood.env.blood_env import BloodMahjongEnv
        cfg = _make_smoke_cfg()
        env = BloodMahjongEnv(cfg)
        obs_dict, info = env.reset(seed=42)

        assert "obs" in obs_dict
        assert "action_mask" in obs_dict
        assert obs_dict["obs"].shape == (OBS_SIZE,)
        assert obs_dict["action_mask"].shape == (ACTION_SPACE,)
        # At least one action should be legal after reset
        assert obs_dict["action_mask"].sum() > 0

    @requires_rust
    def test_env_step(self):
        """Environment should accept an action and return valid results."""
        from blood.env.blood_env import BloodMahjongEnv
        cfg = _make_smoke_cfg()
        env = BloodMahjongEnv(cfg)
        obs_dict, _ = env.reset(seed=42)

        # Pick a legal action
        mask = obs_dict["action_mask"]
        legal_actions = np.where(mask > 0.5)[0]
        assert len(legal_actions) > 0
        action = legal_actions[0]

        obs_dict, reward, terminated, truncated, info = env.step(action)
        assert obs_dict["obs"].shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)


# ══════════════════════════════════════════════════════════════════════════════
# Test 3: Forward Pass
# ══════════════════════════════════════════════════════════════════════════════

class TestForwardPass:
    """Verify forward pass produces correct output shapes with random data."""

    def test_encoder_forward(self):
        """Encoder forward pass should produce (batch, enc_out_dim) tensor."""
        enc = _make_smoke_encoder()
        obs = _random_obs()
        out = enc({"obs": obs})
        assert out.shape == (SMOKE_BATCH, SMOKE_ENC_OUT)

    def test_encoder_output_finite(self):
        """Encoder output should contain only finite values."""
        enc = _make_smoke_encoder()
        obs = _random_obs()
        out = enc({"obs": obs})
        assert torch.isfinite(out).all(), "Encoder produced non-finite values"

    def test_policy_model_forward(self):
        """PolicyModel forward should produce (batch, ACTION_DIM) logits."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        obs = _random_obs()
        logits, values = model(obs)
        assert logits.shape == (SMOKE_BATCH, 34)
        assert values.shape == (SMOKE_BATCH, 1)

    def test_aux_head_forward(self):
        """AuxHead should produce shanten logits (B,3,5) and ow logits (B,81)."""
        head = AuxHead(in_dim=SMOKE_ENC_OUT, hidden=64)
        features = torch.randn(SMOKE_BATCH, SMOKE_ENC_OUT)
        shanten_logits, ow_logits = head(features)
        assert shanten_logits.shape == (SMOKE_BATCH, 3, 5)
        assert ow_logits.shape == (SMOKE_BATCH, 81)


# ══════════════════════════════════════════════════════════════════════════════
# Test 4: Loss Computation
# ══════════════════════════════════════════════════════════════════════════════

class TestLossComputation:
    """Verify losses compute correctly and produce finite scalar values."""

    def test_policy_loss(self):
        """Cross-entropy policy loss on random logits should be finite."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        obs = _random_obs()
        logits, _ = model(obs)

        # Simulate policy loss: CE against random target actions
        targets = torch.randint(0, 34, (SMOKE_BATCH,))
        loss = nn.functional.cross_entropy(logits, targets)
        assert loss.ndim == 0
        assert loss.isfinite(), f"Policy loss is not finite: {loss.item()}"

    def test_value_loss(self):
        """MSE value loss should be finite."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        obs = _random_obs()
        _, values = model(obs)

        targets = torch.randn(SMOKE_BATCH, 1)
        loss = nn.functional.mse_loss(values, targets)
        assert loss.isfinite(), f"Value loss is not finite: {loss.item()}"

    def test_aux_loss(self):
        """AuxHead loss (shanten + opponent waits) should be finite and positive."""
        head = AuxHead(in_dim=SMOKE_ENC_OUT, hidden=64)
        features = torch.randn(SMOKE_BATCH, SMOKE_ENC_OUT)

        shanten_labels = torch.zeros(SMOKE_BATCH, 3, 5)
        shanten_labels[:, :, 0] = 1.0  # all tenpai
        ow_labels = torch.zeros(SMOKE_BATCH, 81)

        loss = head.loss(features, shanten_labels, ow_labels)
        assert loss.ndim == 0
        assert loss.isfinite(), f"Aux loss is not finite: {loss.item()}"
        assert loss.item() > 0

    def test_combined_loss_finite(self):
        """Combined policy + value + aux loss should be finite."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        aux_head = AuxHead(in_dim=SMOKE_ENC_OUT, hidden=64)

        obs = _random_obs()
        logits, values = model(obs)

        # Policy loss
        targets = torch.randint(0, 34, (SMOKE_BATCH,))
        policy_loss = nn.functional.cross_entropy(logits, targets)

        # Value loss
        value_targets = torch.randn(SMOKE_BATCH, 1)
        value_loss = nn.functional.mse_loss(values, value_targets)

        # Aux loss — use encoder features (simulate post-encoder features)
        enc = _make_smoke_encoder()
        features = enc({"obs": obs})
        shanten_labels = torch.zeros(SMOKE_BATCH, 3, 5)
        shanten_labels[:, :, 2] = 1.0
        ow_labels = torch.zeros(SMOKE_BATCH, 81)
        aux_loss = aux_head.loss(features.detach(), shanten_labels, ow_labels)

        total = policy_loss + value_loss + 0.1 * aux_loss
        assert total.isfinite(), f"Combined loss is not finite: {total.item()}"


# ══════════════════════════════════════════════════════════════════════════════
# Test 5: Gradient Update
# ══════════════════════════════════════════════════════════════════════════════

class TestGradientUpdate:
    """Verify a complete forward → backward → optimizer.step cycle works
    and actually changes model parameters."""

    def test_single_gradient_step(self):
        """One forward+backward+step should change at least some parameters."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Snapshot parameters before update
        params_before = {
            name: p.clone().detach()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Forward
        obs = _random_obs()
        logits, values = model(obs)
        targets = torch.randint(0, 34, (SMOKE_BATCH,))
        loss = nn.functional.cross_entropy(logits, targets)

        # Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # At least some parameters should have changed
        any_changed = False
        for name, p in model.named_parameters():
            if p.requires_grad and name in params_before:
                if not torch.allclose(p.data, params_before[name], atol=1e-10):
                    any_changed = True
                    break
        assert any_changed, "No parameters changed after gradient step"

    def test_gradient_norms_finite(self):
        """All gradient norms should be finite after backward pass."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )

        obs = _random_obs()
        logits, values = model(obs)
        targets = torch.randint(0, 34, (SMOKE_BATCH,))
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), (
                    f"Non-finite gradient in {name}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Test 6: Training Loop (10 steps)
# ══════════════════════════════════════════════════════════════════════════════

class TestTrainingLoop:
    """Run a mini training loop to verify loss trends and numerical stability."""

    def test_training_loop_10_steps(self):
        """10-step training loop: loss should not be NaN and should vary."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for step in range(10):
            obs = _random_obs()
            logits, values = model(obs)

            # Policy loss
            targets = torch.randint(0, 34, (SMOKE_BATCH,))
            policy_loss = nn.functional.cross_entropy(logits, targets)

            # Value loss
            value_targets = torch.randn(SMOKE_BATCH, 1)
            value_loss = nn.functional.mse_loss(values, value_targets)

            total_loss = policy_loss + value_loss

            optimizer.zero_grad()
            total_loss.backward()
            # Gradient clipping (matches training config)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(total_loss.item())

        # All losses should be finite (not NaN or Inf)
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"Loss at step {i} is not finite: {l}"

        # Losses should not all be identical (model is learning something)
        unique_losses = set(round(l, 6) for l in losses)
        assert len(unique_losses) > 1, (
            f"All 10 losses are identical ({losses[0]}); model may not be updating"
        )

    def test_training_loop_loss_not_exploding(self):
        """Loss should stay bounded over 10 steps (no gradient explosion)."""
        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for step in range(10):
            obs = _random_obs()
            logits, values = model(obs)
            targets = torch.randint(0, 34, (SMOKE_BATCH,))
            loss = nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Policy CE loss on 34 classes should stay well under 100
            assert loss.item() < 100, (
                f"Loss exploded at step {step}: {loss.item()}"
            )

    @requires_rust
    def test_training_loop_with_env_data(self):
        """10-step loop using real env observations (requires Rust engine)."""
        from blood.env.blood_env import BloodMahjongEnv

        cfg = _make_smoke_cfg()
        env = BloodMahjongEnv(cfg)

        model = PolicyModel(
            obs_channels=SMOKE_OBS_CHANNELS,
            conv_ch=SMOKE_CONV_CH,
            num_blocks=SMOKE_NUM_BLOCKS,
            enc_out_dim=SMOKE_ENC_OUT,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs_dict, _ = env.reset(seed=42)
        losses = []

        for step in range(10):
            obs_t = torch.as_tensor(obs_dict["obs"], dtype=torch.float32).unsqueeze(0)
            mask_t = torch.as_tensor(obs_dict["action_mask"], dtype=torch.float32).unsqueeze(0)

            logits, values = model(obs_t)

            # Mask illegal actions for loss computation
            masked_logits = logits.clone()
            masked_logits[mask_t < 0.5] = float("-inf")

            # Use the most likely legal action as pseudo-target
            legal_actions = np.where(obs_dict["action_mask"] > 0.5)[0]
            if len(legal_actions) == 0:
                legal_actions = np.array([0])
            target = torch.tensor([legal_actions[0]], dtype=torch.long)

            loss = nn.functional.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

            # Step the environment
            action = legal_actions[0]
            obs_dict, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                obs_dict, _ = env.reset(seed=42 + step)

        # All losses should be finite
        for i, l in enumerate(losses):
            assert math.isfinite(l), f"Loss at step {i} is not finite: {l}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
