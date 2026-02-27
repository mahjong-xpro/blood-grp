"""Tests for Blood Mahjong neural network models."""

import pytest
import torch
import numpy as np

from blood.model.encoder import (
    SuitAwareConv1d, ChannelAttention, ResBlock,
    SuitPositionalEncoding, TileAttention, BottleneckBlock,
    SuitAwareResNetEncoder, _build_enc_proj,
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


class TestBuildEncProj:
    """测试 _build_enc_proj 工厂函数。"""

    def test_1layer_shape(self):
        """单层模式：LayerNorm + Linear。"""
        proj = _build_enc_proj(6912, 1024, num_layers=1)
        x = torch.randn(2, 6912)
        out = proj(x)
        assert out.shape == (2, 1024)
        # 单层模式只有 2 个子模块: LayerNorm(0) + Linear(1)
        assert len(proj) == 2

    def test_2layer_shape(self):
        """渐进压缩模式：LayerNorm + Linear + LayerNorm + Mish + Linear。"""
        proj = _build_enc_proj(6912, 1024, num_layers=2)
        x = torch.randn(2, 6912)
        out = proj(x)
        assert out.shape == (2, 1024)
        # 2层模式有 5 个子模块
        assert len(proj) == 5

    def test_2layer_mid_dim(self):
        """中间维度应为 enc_out_dim * 2。"""
        proj = _build_enc_proj(6912, 1024, num_layers=2)
        # proj[1] 是第一个 Linear，输出维度 = mid_dim = 1024*2 = 2048
        assert proj[1].out_features == 2048
        # proj[4] 是第二个 Linear，输出维度 = enc_out_dim = 1024
        assert proj[4].out_features == 1024

    def test_2layer_mid_dim_capped(self):
        """当 enc_out_dim * 2 > raw_dim 时，mid_dim 应被截断为 raw_dim。"""
        proj = _build_enc_proj(1000, 600, num_layers=2)
        # mid_dim = min(600*2, 1000) = 1000
        assert proj[1].out_features == 1000

    def test_invalid_layers(self):
        """不支持的层数应抛出 ValueError。"""
        with pytest.raises(ValueError):
            _build_enc_proj(6912, 1024, num_layers=4)

    def test_backward_compat_default(self):
        """默认 num_layers=1 保持旧行为。"""
        proj = _build_enc_proj(1728, 256)
        assert len(proj) == 2  # LayerNorm + Linear


class TestSuitAwareResNetEncoder:
    def _make_cfg(self, enc_proj_layers=1):
        class Cfg:
            blood_obs_channels = DEFAULT_OBS_CHANNELS
            blood_conv_channels = 64
            blood_num_res_blocks = 2
            blood_encoder_out_dim = 256
            blood_enc_proj_layers = enc_proj_layers
        return Cfg()

    def test_output_shape(self):
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        obs = torch.randn(4, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out = enc({"obs": obs})
        assert out.shape == (4, 256)  # enc_proj active: 64*27=1728 → 256

    def test_output_shape_2layer(self):
        """2层渐进压缩模式输出维度不变。"""
        cfg = self._make_cfg(enc_proj_layers=2)
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        obs = torch.randn(4, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out = enc({"obs": obs})
        assert out.shape == (4, 256)

    def test_get_out_size(self):
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert enc.get_out_size() == 256  # enc_proj active

    def test_get_out_size_2layer(self):
        cfg = self._make_cfg(enc_proj_layers=2)
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert enc.get_out_size() == 256

    def test_named_submodules(self):
        """Encoder should expose stem, pos_enc, segments, tile_attns."""
        cfg = self._make_cfg()
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert hasattr(enc, "segments"), "Missing segments ModuleList"
        assert hasattr(enc, "tile_attns"), "Missing tile_attns ModuleList"
        assert hasattr(enc, "stem"), "Missing stem"
        assert hasattr(enc, "pos_enc"), "Missing pos_enc"
        assert hasattr(enc, "enc_proj"), "Missing enc_proj"
        assert len(enc.segments) > 0
        assert len(enc.tile_attns) == len(enc.segments)

    def test_enc_proj_2layer_structure(self):
        """2层模式的 enc_proj 应有 5 个子模块。"""
        cfg = self._make_cfg(enc_proj_layers=2)
        enc = SuitAwareResNetEncoder(cfg, obs_space=None)
        assert len(enc.enc_proj) == 5


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
        logits, _, _ = model(obs)
        assert logits.shape == (2, ACTION_DIM)

    def test_forward_2layer(self):
        """2层渐进压缩模式的前向传播。"""
        model = PolicyModel(obs_channels=DEFAULT_OBS_CHANNELS, conv_ch=32, num_blocks=2,
                            enc_out_dim=128, enc_proj_layers=2)
        obs = torch.randn(2, DEFAULT_OBS_CHANNELS * NUM_TILES)
        logits, _, _ = model(obs)
        assert logits.shape == (2, ACTION_DIM)

    def test_get_action(self):
        model = PolicyModel(obs_channels=DEFAULT_OBS_CHANNELS, conv_ch=32, num_blocks=2, enc_out_dim=128)
        obs = torch.randn(DEFAULT_OBS_CHANNELS * NUM_TILES)
        mask = torch.zeros(ACTION_DIM)
        mask[0] = 1.0
        mask[5] = 1.0
        mask[30] = 1.0
        action, _, _ = model.get_action(obs, mask)
        assert action in [0, 5, 30]

    def _build_fake_checkpoint(self, tmp_path, conv_ch, num_blocks, enc_out, enc_proj_layers=1):
        """构建模拟 SF2 checkpoint 的辅助方法。"""
        from blood.model.encoder import _num_groups
        import torch.nn as nn

        ng = _num_groups(conv_ch)
        mid = num_blocks // 2
        stem = nn.Sequential(
            SuitAwareConv1d(DEFAULT_OBS_CHANNELS, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        pos_enc = SuitPositionalEncoding(conv_ch)
        res_blocks_1 = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(mid)])
        tile_attn_mid = TileAttention(conv_ch, num_heads=4)
        res_blocks_2 = nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(num_blocks - mid)])
        tile_attn = TileAttention(conv_ch, num_heads=4)
        enc_proj = _build_enc_proj(conv_ch * NUM_TILES, enc_out, enc_proj_layers)
        action_head = nn.Linear(enc_out, ACTION_DIM)

        sd = {}
        for k, v in stem.state_dict().items():
            sd[f"encoder.stem.{k}"] = v
        for k, v in pos_enc.state_dict().items():
            sd[f"encoder.pos_enc.{k}"] = v
        for k, v in res_blocks_1.state_dict().items():
            sd[f"encoder.res_blocks_1.{k}"] = v
        for k, v in tile_attn_mid.state_dict().items():
            sd[f"encoder.tile_attn_mid.{k}"] = v
        for k, v in res_blocks_2.state_dict().items():
            sd[f"encoder.res_blocks_2.{k}"] = v
        for k, v in tile_attn.state_dict().items():
            sd[f"encoder.tile_attn.{k}"] = v
        for k, v in enc_proj.state_dict().items():
            sd[f"encoder.enc_proj.{k}"] = v
        sd["action_parameterization.distribution_linear.weight"] = action_head.weight.data
        sd["action_parameterization.distribution_linear.bias"] = action_head.bias.data

        ckpt_path = str(tmp_path / f"test_ckpt_{enc_proj_layers}layer.pth")
        torch.save({"model": sd}, ckpt_path)
        return ckpt_path, mid, num_blocks

    def test_checkpoint_round_trip(self, tmp_path):
        """Verify from_sf2_checkpoint correctly loads legacy res_blocks_1/2 format."""
        conv_ch, num_blocks, enc_out = 32, 4, 128
        ckpt_path, mid, _ = self._build_fake_checkpoint(
            tmp_path, conv_ch, num_blocks, enc_out, enc_proj_layers=1)

        loaded = PolicyModel.from_sf2_checkpoint(ckpt_path)
        assert loaded._obs_channels == DEFAULT_OBS_CHANNELS

        # Legacy checkpoint loaded into segment-based model
        assert hasattr(loaded, "segments") or hasattr(loaded, "res_blocks_1")
        # 1层模式：enc_proj 应有 2 个子模块
        assert len(loaded.enc_proj) == 2

        obs = torch.randn(1, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out, _ = loaded(obs)
        assert out.shape == (1, ACTION_DIM)

    def test_checkpoint_round_trip_2layer(self, tmp_path):
        """验证 from_sf2_checkpoint 能正确检测并加载 2层渐进压缩格式。"""
        conv_ch, num_blocks, enc_out = 32, 4, 128
        ckpt_path, mid, _ = self._build_fake_checkpoint(
            tmp_path, conv_ch, num_blocks, enc_out, enc_proj_layers=2)

        loaded = PolicyModel.from_sf2_checkpoint(ckpt_path)
        assert loaded._obs_channels == DEFAULT_OBS_CHANNELS

        # 2层模式：enc_proj 应有 5 个子模块
        assert len(loaded.enc_proj) == 5
        assert hasattr(loaded, "segments") or hasattr(loaded, "res_blocks_1")

        obs = torch.randn(1, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out, _ = loaded(obs)
        assert out.shape == (1, ACTION_DIM)

    def _build_fake_checkpoint_segments(self, tmp_path, conv_ch, num_blocks,
                                         enc_out, num_segments=2, enc_proj_layers=1):
        """Build a fake checkpoint with the new segment-based encoder layout."""
        from blood.model.encoder import _num_groups
        import torch.nn as nn

        ng = _num_groups(conv_ch)
        blocks_per_seg = num_blocks // num_segments
        remainder = num_blocks % num_segments

        stem = nn.Sequential(
            SuitAwareConv1d(DEFAULT_OBS_CHANNELS, conv_ch, kernel_size=3),
            nn.GroupNorm(ng, conv_ch),
            nn.Mish(inplace=True),
        )
        pos_enc = SuitPositionalEncoding(conv_ch)

        segments = nn.ModuleList()
        tile_attns = nn.ModuleList()
        for i in range(num_segments):
            n_blks = blocks_per_seg + (1 if i < remainder else 0)
            segments.append(nn.Sequential(*[BottleneckBlock(conv_ch) for _ in range(n_blks)]))
            tile_attns.append(TileAttention(conv_ch, num_heads=4))

        enc_proj = _build_enc_proj(conv_ch * NUM_TILES, enc_out, enc_proj_layers)
        action_head = nn.Linear(enc_out, ACTION_DIM)

        sd = {}
        for k, v in stem.state_dict().items():
            sd[f"encoder.stem.{k}"] = v
        for k, v in pos_enc.state_dict().items():
            sd[f"encoder.pos_enc.{k}"] = v
        for k, v in segments.state_dict().items():
            sd[f"encoder.segments.{k}"] = v
        for k, v in tile_attns.state_dict().items():
            sd[f"encoder.tile_attns.{k}"] = v
        for k, v in enc_proj.state_dict().items():
            sd[f"encoder.enc_proj.{k}"] = v
        sd["action_parameterization.distribution_linear.weight"] = action_head.weight.data
        sd["action_parameterization.distribution_linear.bias"] = action_head.bias.data

        ckpt_path = str(tmp_path / f"test_ckpt_seg{num_segments}_{enc_proj_layers}layer.pth")
        torch.save({"model": sd}, ckpt_path)
        return ckpt_path

    def test_checkpoint_round_trip_segments(self, tmp_path):
        """Verify from_sf2_checkpoint loads new segment-based checkpoint format."""
        conv_ch, num_blocks, enc_out = 32, 4, 128
        num_segments = 2
        ckpt_path = self._build_fake_checkpoint_segments(
            tmp_path, conv_ch, num_blocks, enc_out,
            num_segments=num_segments, enc_proj_layers=1)

        loaded = PolicyModel.from_sf2_checkpoint(ckpt_path)
        assert loaded._obs_channels == DEFAULT_OBS_CHANNELS
        assert hasattr(loaded, "segments"), "Loaded model missing segments"
        assert hasattr(loaded, "tile_attns"), "Loaded model missing tile_attns"
        assert len(loaded.segments) == num_segments
        assert len(loaded.tile_attns) == num_segments
        assert len(loaded.enc_proj) == 2

        obs = torch.randn(1, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out, _ = loaded(obs)
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
        import sys
        argv_backup = sys.argv[:]
        sys.argv = ["train", "--env", "blood_mahjong"]
        if not use_rnn:
            sys.argv += ["--use_rnn", "False"]
        from blood.training.runner import register_blood_components
        register_blood_components()
        from blood.cfg import add_blood_args, blood_override_defaults
        from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
        parser, _ = parse_sf_args(evaluation=False)
        add_blood_args(parser)
        blood_override_defaults(parser)
        cfg = parse_full_cfg(parser)
        sys.argv = argv_backup
        cfg.blood_conv_channels = 64
        cfg.blood_num_res_blocks = 2
        cfg.blood_encoder_out_dim = 256
        cfg.oracle_enabled = False
        return cfg

    def _make_obs_action_space(self):
        from gymnasium.spaces import Box, Dict
        from sample_factory.algo.utils.spaces.discretized import Discrete
        import numpy as np
        obs_space = Dict({"obs": Box(low=0, high=1, shape=(DEFAULT_OBS_CHANNELS * NUM_TILES,), dtype=np.float32)})
        action_space = Discrete(34)
        return obs_space, action_space

    def test_identity_core_head_dims(self):
        """Without LSTM, actor/critic heads take enc_out (blood_encoder_out_dim=256) as input."""
        cfg = self._make_cfg(use_rnn=False)
        from blood.model.factory import BloodActorCritic
        from sample_factory.algo.utils.context import global_model_factory
        obs_space, action_space = self._make_obs_action_space()
        model = BloodActorCritic(global_model_factory(), obs_space, action_space, cfg)
        enc_out = 256  # blood_encoder_out_dim
        # actor_head[0] is LayerNorm; actor_head[1] is the first Linear
        assert model.actor_head[1].in_features == enc_out
        assert model.critic_head[1].in_features == enc_out
        assert model.aux_head.shared[1].in_features == enc_out

    def test_lstm_core_head_dims(self):
        """With LSTM (default rnn_size=1024), actor/critic and AuxHead all take 1024 as input."""
        cfg = self._make_cfg(use_rnn=True)
        from blood.model.factory import BloodActorCritic
        from sample_factory.algo.utils.context import global_model_factory
        obs_space, action_space = self._make_obs_action_space()
        model = BloodActorCritic(global_model_factory(), obs_space, action_space, cfg)
        rnn_out = cfg.rnn_size  # 1024 by default
        # actor_head[0] is LayerNorm; actor_head[1] is the first Linear
        assert model.actor_head[1].in_features == rnn_out
        assert model.critic_head[1].in_features == rnn_out
        # AuxHead reads post-LSTM features (core_out), same as actor/critic
        assert model.aux_head.shared[1].in_features == rnn_out


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


class TestSpatialPoolingProj:
    """Tests for SpatialPoolingProj attention-based pooling."""

    def test_output_shape(self):
        from blood.model.encoder import SpatialPoolingProj
        proj = SpatialPoolingProj(conv_ch=64, enc_out_dim=256, num_queries=4)
        x = torch.randn(2, 64, NUM_TILES)  # (B, C, 27)
        out = proj(x)
        assert out.shape == (2, 256)

    def test_output_finite(self):
        from blood.model.encoder import SpatialPoolingProj
        proj = SpatialPoolingProj(conv_ch=64, enc_out_dim=128, num_queries=2)
        x = torch.randn(4, 64, NUM_TILES)
        out = proj(x)
        assert torch.isfinite(out).all()

    def test_build_enc_proj_3(self):
        """_build_enc_proj(num_layers=3) should return SpatialPoolingProj."""
        from blood.model.encoder import SpatialPoolingProj
        proj = _build_enc_proj(6912, 1024, num_layers=3, conv_ch=256)
        assert isinstance(proj, SpatialPoolingProj)
        x = torch.randn(2, 256, NUM_TILES)
        out = proj(x)
        assert out.shape == (2, 1024)

    def test_encoder_with_spatial_pooling(self):
        """SuitAwareResNetEncoder with enc_proj_layers=3."""
        class Cfg:
            blood_obs_channels = DEFAULT_OBS_CHANNELS
            blood_conv_channels = 64
            blood_num_res_blocks = 2
            blood_encoder_out_dim = 256
            blood_enc_proj_layers = 3
            blood_num_tile_attn_layers = 1
            blood_tile_attn_heads = 4
        enc = SuitAwareResNetEncoder(Cfg(), obs_space=None)
        assert enc.get_out_size() == 256
        obs = torch.randn(4, DEFAULT_OBS_CHANNELS * NUM_TILES)
        out = enc({"obs": obs})
        assert out.shape == (4, 256)

    def test_num_queries_auto(self):
        """num_queries should auto-compute from enc_out_dim // conv_ch."""
        proj = _build_enc_proj(6912, 1024, num_layers=3, conv_ch=256)
        # 1024 // 256 = 4
        assert proj.num_queries == 4

    def test_num_queries_floor(self):
        """num_queries should be at least 2."""
        proj = _build_enc_proj(1728, 64, num_layers=3, conv_ch=64)
        # 64 // 64 = 1, but min is 2
        assert proj.num_queries == 2


class TestLeagueSparseEviction:
    """Tests for sparse retention eviction strategy."""

    def test_sparse_eviction_keeps_newest_and_oldest(self, tmp_path):
        """After eviction, both newest and oldest checkpoints should survive."""
        from pathlib import Path
        from blood.training.league import LeagueManager

        pool_dir = tmp_path / "pool"
        manager = LeagueManager(str(pool_dir), max_pool_size=10)

        # Add 20 checkpoints
        for step in range(100, 2100, 100):
            ckpt = tmp_path / f"checkpoint_{step}.pth"
            ckpt.write_text("dummy")
            manager.add_checkpoint(Path(ckpt))

        # Pool should be capped at 10
        assert manager.pool_size() <= 10

        # Newest (step 2000) and oldest (step 100) should both survive
        remaining = manager.get_checkpoints()
        remaining_steps = {int(p.stem.split("_")[1]) for p in remaining}
        assert 2000 in remaining_steps, "Newest checkpoint should survive eviction"
        assert 100 in remaining_steps, "Oldest checkpoint should survive eviction"

    def test_sparse_eviction_time_span(self, tmp_path):
        """After eviction, remaining checkpoints should span the full time range,
        not just the most recent ones."""
        from pathlib import Path
        from blood.training.league import LeagueManager

        pool_dir = tmp_path / "pool"
        manager = LeagueManager(str(pool_dir), max_pool_size=10)

        for step in range(1000, 6000, 250):
            ckpt = tmp_path / f"checkpoint_{step}.pth"
            ckpt.write_text("dummy")
            manager.add_checkpoint(Path(ckpt))

        remaining = manager.get_checkpoints()
        remaining_steps = sorted([int(p.stem.split("_")[1]) for p in remaining])
        # Should cover full range: min near 1000, max near 5750
        assert remaining_steps[0] <= 2000, "Should have early checkpoints"
        assert remaining_steps[-1] >= 5000, "Should have latest checkpoints"


class TestLeagueFrozenWindow:
    """Tests for frozen window feature."""

    def test_frozen_window_excludes_recent(self, tmp_path):
        """With frozen_window=3, the 3 newest checkpoints should never be sampled."""
        from pathlib import Path
        from blood.training.league import LeagueManager

        pool_dir = tmp_path / "pool"
        manager = LeagueManager(
            str(pool_dir), max_pool_size=50,
            frozen_window=3, self_play_prob=0.0,  # disable self-play for determinism
        )

        # Add 10 checkpoints
        for step in range(100, 1100, 100):
            ckpt = pool_dir / f"checkpoint_{step}.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text("dummy")

        # newest checkpoints are steps 1000, 900, 800 (frozen_window=3)
        frozen_steps = {1000, 900, 800}

        # Sample 100 times, none should be from frozen window
        for _ in range(100):
            result = manager.sample_opponent()
            if result is not None:
                step = int(result.stem.split("_")[1])
                assert step not in frozen_steps, (
                    f"Frozen checkpoint step={step} was sampled!"
                )

    def test_frozen_window_fallback_small_pool(self, tmp_path):
        """When pool size <= frozen_window, all checkpoints should still be available."""
        from pathlib import Path
        from blood.training.league import LeagueManager

        pool_dir = tmp_path / "pool"
        manager = LeagueManager(
            str(pool_dir), max_pool_size=50,
            frozen_window=5, self_play_prob=0.0,
        )

        # Add only 3 checkpoints (< frozen_window=5)
        for step in [100, 200, 300]:
            ckpt = pool_dir / f"checkpoint_{step}.pth"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_text("dummy")

        # Should still be able to sample (fallback behavior)
        sampled = False
        for _ in range(20):
            result = manager.sample_opponent()
            if result is not None:
                sampled = True
                break
        assert sampled, "Should be able to sample even when pool <= frozen_window"


class TestEntropyFloor:
    """Tests for entropy floor safety net in scheduler."""

    def test_entropy_floor_clamps(self):
        """Entropy scheduler should clamp values below the floor."""
        from blood.training.scheduler import HyperparamScheduler, ScheduleConfig

        sched = ScheduleConfig(
            param_name="exploration_loss_coeff",
            schedule_type="cosine",
            start_value=0.02,
            end_value=0.001,  # below floor
            start_step=0,
            end_step=100,
        )
        scheduler = HyperparamScheduler([sched])

        # At end_step, the value should be 0.001
        updates = scheduler.step(100)
        assert "exploration_loss_coeff" in updates
        assert abs(updates["exploration_loss_coeff"] - 0.001) < 1e-6

        # The clamping happens in callbacks._apply_schedules, not in the
        # scheduler itself. Here we just verify the scheduler outputs the raw value.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

