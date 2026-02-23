"""Tests for Blood Mahjong neural network models."""

import pytest
import torch
import numpy as np

from blood.model.encoder import (
    SuitAwareConv1d, ChannelAttention, ResBlock,
    SuitPositionalEncoding, TileAttention, BottleneckBlock,
    SuitAwareResNetEncoder,
    NUM_TILES, DEFAULT_OBS_CHANNELS,
)
from blood.model.heads import AuxHead
from blood.model.oracle import OracleEncoder, DistillationLoss
from blood.model.inference import PolicyModel

ACTION_DIM = 34


class TestSuitPositionalEncoding:
    def test_output_shape(self):
        pe = SuitPositionalEncoding(32)
        x = torch.randn(2, 32, NUM_TILES)
        out = pe(x)
        assert out.shape == (2, 32, NUM_TILES)

    def test_suit_shared_offset(self):
        """Man and Pin should receive the same positional offset."""
        pe = SuitPositionalEncoding(8)
        x = torch.zeros(1, 8, NUM_TILES)
        out = pe(x)
        # The added embedding for Man (0:9) and Pin (9:18) should be identical
        assert torch.allclose(out[:, :, 0:9], out[:, :, 9:18], atol=1e-6)
        assert torch.allclose(out[:, :, 9:18], out[:, :, 18:27], atol=1e-6)


class TestTileAttention:
    def test_output_shape(self):
        attn = TileAttention(64, num_heads=4)
        x = torch.randn(2, 64, NUM_TILES)
        out = attn(x)
        assert out.shape == (2, 64, NUM_TILES)

    def test_residual_connection(self):
        """Output should differ from input (attention contributes)."""
        attn = TileAttention(32, num_heads=4)
        x = torch.randn(2, 32, NUM_TILES)
        out = attn(x)
        assert not torch.allclose(out, x)


class TestSuitAwareConv1d:
    def test_output_shape(self):
        conv = SuitAwareConv1d(16, 32, kernel_size=3)
        x = torch.randn(2, 16, NUM_TILES)
        out = conv(x)
        assert out.shape == (2, 32, NUM_TILES)

    def test_shared_weights(self):
        """All three suits should use the same conv kernel."""
        conv = SuitAwareConv1d(8, 16)
        x = torch.zeros(1, 8, NUM_TILES)
        x[:, :, 0:9] = 1.0  # Man
        x[:, :, 9:18] = 1.0  # Pin (same input pattern)
        out = conv(x)
        # Man and Pin outputs should be identical (shared weights, same input)
        assert torch.allclose(out[:, :, 0:9], out[:, :, 9:18], atol=1e-5)


class TestChannelAttention:
    def test_output_shape(self):
        attn = ChannelAttention(64)
        x = torch.randn(4, 64, NUM_TILES)
        out = attn(x)
        assert out.shape == x.shape

    def test_scale_range(self):
        attn = ChannelAttention(32)
        x = torch.randn(2, 32, NUM_TILES)
        out = attn(x)
        # Sigmoid-scaled output should generally not explode
        assert out.abs().max() < 100


class TestResBlock:
    def test_residual_connection(self):
        block = ResBlock(64)
        x = torch.randn(2, 64, NUM_TILES)
        out = block(x)
        assert out.shape == x.shape
        # Output should be different from input (residual branch contributes)
        assert not torch.allclose(out, x)


class TestSuitAwareResNetEncoder:
    def _make_cfg(self):
        class Cfg:
            blood_obs_channels = DEFAULT_OBS_CHANNELS
            blood_conv_channels = 64
            blood_num_res_blocks = 2
            blood_encoder_out_dim = 256
        return Cfg()

    def test_output_shape(self):
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        obs = torch.randn(4, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out = enc({"obs": obs})
        assert out.shape == (4, 256)  # enc_proj active: 64*27=1728 → 256

    def test_get_out_size(self):
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert enc.get_out_size() == 256  # enc_proj active

    def test_named_submodules(self):
        """Encoder should expose stem, pos_enc, res_blocks_1/2, tile_attn_mid/tile_attn."""
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert hasattr(enc, "stem")
        assert hasattr(enc, "pos_enc")
        assert hasattr(enc, "res_blocks_1")
        assert hasattr(enc, "res_blocks_2")
        assert hasattr(enc, "tile_attn_mid")
        assert hasattr(enc, "tile_attn")


class TestAuxHead:
    def test_forward_shape(self):
        head = AuxHead(in_dim=256, hidden=128)
        features = torch.randn(8, 256)
        shanten_logits, ow_logits = head(features)
        assert shanten_logits.shape == (8, 3, 5)
        assert ow_logits.shape == (8, 81)

    def test_loss_computes(self):
        head = AuxHead(in_dim=256, hidden=128)
        features = torch.randn(8, 256)
        shanten_labels = torch.zeros(8, 3, 5)
        shanten_labels[:, :, 0] = 1.0  # all tenpai
        ow_labels = torch.zeros(8, 81)
        loss = head.loss(features, shanten_labels, ow_labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_shanten_classes(self):
        """All 5 shanten classes should produce finite loss."""
        head = AuxHead(in_dim=256, hidden=128)
        features = torch.randn(4, 256)
        for cls in range(5):
            shanten_labels = torch.zeros(4, 3, 5)
            shanten_labels[:, :, cls] = 1.0
            ow_labels = torch.zeros(4, 81)
            loss = head.loss(features, shanten_labels, ow_labels)
            assert loss.isfinite()


class TestOracleEncoder:
    def test_output_shape(self):
        oracle = OracleEncoder(obs_channels=516, conv_ch=32, num_blocks=2, action_dim=34)
        obs = torch.randn(4, 516 * NUM_TILES)
        logits, values = oracle(obs)
        assert logits.shape == (4, 34)
        assert values.shape == (4, 1)


class TestDistillationLoss:
    def test_basic_loss(self):
        loss_fn = DistillationLoss(temperature=2.0)
        student = torch.randn(8, 34)
        oracle = torch.randn(8, 34)
        loss = loss_fn(student, oracle)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_masked_loss(self):
        loss_fn = DistillationLoss(temperature=2.0)
        student = torch.randn(4, 34)
        oracle = torch.randn(4, 34)
        mask = torch.zeros(4, 34, dtype=torch.bool)
        mask[:, :10] = True
        loss = loss_fn(student, oracle, action_mask=mask)
        assert loss.isfinite()


class TestPolicyModel:
    def test_forward(self):
        model = PolicyModel(obs_channels=DEFAULT_OBS_CHANNELS, conv_ch=32, num_blocks=2, enc_out_dim=128)
        obs = torch.randn(2, DEFAULT_OBS_CHANNELS * NUM_TILES)
        logits, _ = model(obs)
        assert logits.shape == (2, ACTION_DIM)

    def test_get_action(self):
        model = PolicyModel(obs_channels=DEFAULT_OBS_CHANNELS, conv_ch=32, num_blocks=2, enc_out_dim=128)
        obs = torch.randn(DEFAULT_OBS_CHANNELS * NUM_TILES)
        mask = torch.zeros(ACTION_DIM)
        mask[0] = 1.0
        mask[5] = 1.0
        mask[30] = 1.0
        action, _ = model.get_action(obs, mask)
        assert action in [0, 5, 30]

    def test_checkpoint_round_trip(self, tmp_path):
        """Verify from_sf2_checkpoint correctly infers architecture params."""
        conv_ch, num_blocks, enc_out = 32, 4, 128
        from blood.model.encoder import _num_groups
        import torch.nn as nn

        # Build a minimal encoder matching the new named-submodule structure
        ng = _num_groups(conv_ch)
        stem = nn.Sequential(
            SuitAwareConv1d(DEFAULT_OBS_CHANNELS, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        pos_enc = SuitPositionalEncoding(conv_ch)
        res_blocks = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(num_blocks)])
        tile_attn = TileAttention(conv_ch, num_heads=4)
        fc = nn.Sequential(nn.Linear(conv_ch * NUM_TILES, enc_out), nn.Mish(inplace=True))
        action_head = nn.Linear(enc_out, ACTION_DIM)

        sd = {}
        for k, v in stem.state_dict().items():
            sd[f"encoder.stem.{k}"] = v
        for k, v in pos_enc.state_dict().items():
            sd[f"encoder.pos_enc.{k}"] = v
        for k, v in res_blocks.state_dict().items():
            sd[f"encoder.res_blocks.{k}"] = v
        for k, v in tile_attn.state_dict().items():
            sd[f"encoder.tile_attn.{k}"] = v
        for k, v in fc.state_dict().items():
            sd[f"encoder.fc.{k}"] = v
        sd["action_parameterization.distribution_linear.weight"] = action_head.weight.data
        sd["action_parameterization.distribution_linear.bias"] = action_head.bias.data

        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({"model": sd}, ckpt_path)

        loaded = PolicyModel.from_sf2_checkpoint(ckpt_path)
        assert loaded._obs_channels == DEFAULT_OBS_CHANNELS

        obs = torch.randn(1, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out = loaded(obs)
        assert out.shape == (1, ACTION_DIM)


class TestRTPA:
    def test_temperature_range(self):
        from blood.eval.rtpa import RTPA
        rtpa = RTPA()
        temp = rtpa.compute_temperature(
            is_tenpai=True, opponents_likely_tenpai=0,
            my_score=100000, avg_opponent_score=100000.0, wall_remaining=50,
        )
        assert 0.3 <= temp <= 3.0

    def test_defense_temperature(self):
        from blood.eval.rtpa import RTPA
        rtpa = RTPA(defend_temp=2.0)
        temp = rtpa.compute_temperature(
            is_tenpai=False, opponents_likely_tenpai=3,
            my_score=100000, avg_opponent_score=100000.0, wall_remaining=50,
        )
        assert temp > 1.0


class TestISMCESearcher:
    def test_select_action(self):
        from blood.eval.ismce import ISMCESearcher
        searcher = ISMCESearcher()
        logits = np.random.randn(ACTION_DIM).astype(np.float32)
        mask = np.zeros(ACTION_DIM, dtype=np.float32)
        mask[0] = 1.0
        mask[30] = 1.0
        action = searcher.select_action(logits, mask)
        assert action in [0, 30]


class TestGameStateTracker:
    def test_update_from_obs(self):
        from blood.eval.rtpa import GameStateTracker, NUM_STUDENT_CHANNELS, NUM_TILE_TYPES
        tracker = GameStateTracker()
        obs = np.zeros(NUM_STUDENT_CHANNELS * NUM_TILE_TYPES, dtype=np.float32)
        tracker.update_from_obs(obs, scores=[70000, 50000, 55000, 65000])
        assert tracker.my_score == 70000
        assert tracker.opponent_scores == [50000, 55000, 65000]

    def test_reset(self):
        from blood.eval.rtpa import GameStateTracker
        tracker = GameStateTracker()
        tracker.my_tenpai = True
        tracker.reset()
        assert tracker.my_tenpai is False


class TestBloodActorCriticDims:
    """Verify actor/critic head input dims adapt to core output (LSTM vs Identity)."""

    def _make_cfg(self, use_rnn=False):
        class Cfg:
            blood_obs_channels = DEFAULT_OBS_CHANNELS
            blood_conv_channels = 64
            blood_num_res_blocks = 2
            blood_encoder_out_dim = 256
            # LSTM
            rnn_type = "lstm"
            rnn_size = 512
            rnn_num_layers = 1
            # Aux / oracle
            aux_shanten_weight = 1.0
            aux_opp_waits_weight = 0.3
            oracle_enabled = False
        cfg = Cfg()
        cfg.use_rnn = use_rnn
        return cfg

    def test_identity_core_head_dims(self):
        """Without LSTM, actor/critic heads take enc_out (64*27=1728) as input."""
        cfg = self._make_cfg(use_rnn=False)
        from blood.model.factory import BloodActorCritic
        from sample_factory.algo.utils.context import global_model_factory
        import gym
        obs_space = {"obs": gym.spaces.Box(low=0, high=1, shape=(DEFAULT_OBS_CHANNELS * NUM_TILES,))}
        action_space = gym.spaces.Discrete(34)
        model = BloodActorCritic(global_model_factory(), obs_space, action_space, cfg)
        enc_out = 64 * NUM_TILES  # 1728
        assert model.actor_head[0].in_features == enc_out
        assert model.critic_head[0].in_features == enc_out
        assert model.aux_head.shared[0].in_features == enc_out

    def test_lstm_core_head_dims(self):
        """With LSTM (rnn_size=512), actor/critic heads take 512 as input."""
        cfg = self._make_cfg(use_rnn=True)
        from blood.model.factory import BloodActorCritic
        from sample_factory.algo.utils.context import global_model_factory
        import gym
        obs_space = {"obs": gym.spaces.Box(low=0, high=1, shape=(DEFAULT_OBS_CHANNELS * NUM_TILES,))}
        action_space = gym.spaces.Discrete(34)
        model = BloodActorCritic(global_model_factory(), obs_space, action_space, cfg)
        enc_out = 64 * NUM_TILES  # 1728
        assert model.actor_head[0].in_features == 512
        assert model.critic_head[0].in_features == 512
        # AuxHead always uses pre-LSTM enc_out
        assert model.aux_head.shared[0].in_features == enc_out


class TestLeagueManager:
    def test_sample_empty(self, tmp_path):
        from blood.training.league import LeagueManager
        manager = LeagueManager(str(tmp_path), newest_weight=3.0)
        assert manager.sample_opponent() is None

    def test_add_and_sample(self, tmp_path):
        from pathlib import Path
        from blood.training.league import LeagueManager
        pool_dir = tmp_path / "pool"
        manager = LeagueManager(str(pool_dir), newest_weight=3.0)
        ckpt = tmp_path / "checkpoint_100.pth"
        ckpt.write_text("dummy")
        manager.add_checkpoint(Path(ckpt))
        assert manager.pool_size() == 1
        result = manager.sample_opponent()
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
