"""Tests for Blood Mahjong Gymnasium environments."""

import pytest
import numpy as np

from blood.env.blood_env import (
    BloodMahjongEnv, OBS_SIZE, ORACLE_OBS_SIZE, ACTION_SPACE,
)

ENGINE_AVAILABLE = False
try:
    from blood._engine import RustMahjongEnv
    ENGINE_AVAILABLE = True
except ImportError:
    pass

needs_engine = pytest.mark.skipif(not ENGINE_AVAILABLE, reason="Rust engine not built")


class TestBloodMahjongEnvFallback:
    """Tests that run without the Rust engine (fallback mode)."""

    def test_create_env(self):
        env = BloodMahjongEnv()
        assert env.observation_space is not None
        assert env.action_space.n == ACTION_SPACE

    def test_observation_space_shape(self):
        env = BloodMahjongEnv()
        assert env.observation_space["obs"].shape == (OBS_SIZE,)
        assert env.observation_space["oracle_obs"].shape == (ORACLE_OBS_SIZE,)
        assert env.observation_space["action_mask"].shape == (ACTION_SPACE,)
        assert env.observation_space["dq_labels"].shape == (3,)
        assert env.observation_space["ow_labels"].shape == (81,)

    def test_reset_returns_valid(self):
        env = BloodMahjongEnv()
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert obs["obs"].shape == (OBS_SIZE,)
        assert obs["action_mask"].shape == (ACTION_SPACE,)

    def test_step_returns_valid(self):
        env = BloodMahjongEnv()
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(30)  # Pass
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)


@needs_engine
class TestBloodMahjongEnvWithEngine:
    """Tests that require the compiled Rust engine."""

    def test_reset_with_engine(self):
        env = BloodMahjongEnv()
        obs, info = env.reset(seed=42)
        mask = obs["action_mask"]
        assert mask.sum() > 0, "Should have at least one legal action"

    def test_full_game_random(self):
        """Play a full game with random legal actions."""
        env = BloodMahjongEnv()
        obs, _ = env.reset(seed=123)
        steps = 0
        max_steps = 500

        while steps < max_steps:
            mask = obs["action_mask"]
            legal = np.where(mask > 0.5)[0]
            assert len(legal) > 0, f"No legal actions at step {steps}"
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            if terminated or truncated:
                break

        assert terminated or truncated or steps == max_steps

    def test_augmentation_consistency(self):
        """Verify augmentation doesn't break observation shapes."""
        env = BloodMahjongEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)

        # Different random states may give different augmentations
        assert obs1["obs"].shape == obs2["obs"].shape
        assert obs1["action_mask"].shape == obs2["action_mask"].shape

    def test_multiple_games(self):
        """Run multiple games to check for crashes."""
        env = BloodMahjongEnv()
        for game in range(10):
            obs, _ = env.reset(seed=game)
            for step in range(100):
                mask = obs["action_mask"]
                legal = np.where(mask > 0.5)[0]
                if len(legal) == 0:
                    break
                action = np.random.choice(legal)
                obs, reward, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    break

    def test_scores_accessible(self):
        env = BloodMahjongEnv()
        env.reset(seed=42)
        scores = env._env.get_scores()
        assert len(scores) == 4
        assert all(s > 0 for s in scores)

    def test_external_mode_api(self):
        """Test the low-level external API methods."""
        rust_env = RustMahjongEnv(42, "external")
        rust_env.reset(42)

        phase = rust_env.get_phase()
        assert phase in ("ding_que", "self_check", "kan_select", "discard", "reaction",
                         "scoring", "done")

        cp = rust_env.get_current_player()
        assert 0 <= cp < 4

        dq_done = rust_env.get_ding_que_done()
        assert len(dq_done) == 4

        obs_dict = rust_env.get_player_obs(0)
        assert "obs" in obs_dict
        assert "action_mask" in obs_dict


@needs_engine
class TestSelfPlayEnv:
    """Tests for the SelfPlayEnv (external opponent mode)."""

    def test_import(self):
        from blood.env.selfplay_env import SelfPlayEnv
        env = SelfPlayEnv()
        assert env.observation_space is not None

    def test_reset_and_step(self):
        from blood.env.selfplay_env import SelfPlayEnv
        env = SelfPlayEnv()
        obs, info = env.reset(seed=42)
        assert obs["obs"].shape == (OBS_SIZE,)

        mask = obs["action_mask"]
        legal = np.where(mask > 0.5)[0]
        if len(legal) > 0:
            obs, reward, term, trunc, info = env.step(int(legal[0]))
            assert obs["obs"].shape == (OBS_SIZE,)

    def test_full_game_selfplay(self):
        """Play through a full game with SelfPlayEnv (fallback random opponents)."""
        from blood.env.selfplay_env import SelfPlayEnv
        env = SelfPlayEnv()
        obs, _ = env.reset(seed=99)
        steps = 0

        while steps < 500:
            mask = obs["action_mask"]
            legal = np.where(mask > 0.5)[0]
            if len(legal) == 0:
                break
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, info = env.step(int(action))
            steps += 1
            if terminated or truncated:
                break

        assert steps > 0


class TestAugmentation:
    """Test suit permutation augmentation correctness."""

    def test_augment_action_round_trip(self):
        from blood.env.augment import augment_action, SUIT_PERMUTATIONS
        for perm in SUIT_PERMUTATIONS[1:]:
            inv_perm = tuple(perm.index(i) for i in range(3))
            for action in range(ACTION_SPACE):
                aug = augment_action(action, perm)
                restored = augment_action(aug, inv_perm)
                assert restored == action, f"perm={perm}, action={action}, aug={aug}, restored={restored}"

    def test_augment_obs_round_trip(self):
        from blood.env.augment import augment_obs, SUIT_PERMUTATIONS
        rng = np.random.default_rng(42)
        obs = rng.random((10, 27)).astype(np.float32)
        for perm in SUIT_PERMUTATIONS[1:]:
            inv_perm = tuple(perm.index(i) for i in range(3))
            aug = augment_obs(obs, perm)
            restored = augment_obs(aug, inv_perm)
            np.testing.assert_allclose(restored, obs, atol=1e-6)

    def test_augment_mask_vectorized(self):
        """Verify the vectorized mask augmentation matches element-wise."""
        from blood.env.augment import augment_action, SUIT_PERMUTATIONS
        rng = np.random.default_rng(7)
        mask = rng.integers(0, 2, size=ACTION_SPACE).astype(np.float32)
        perm = SUIT_PERMUTATIONS[3]

        expected = np.zeros_like(mask)
        for i in range(ACTION_SPACE):
            if mask[i] > 0:
                j = augment_action(i, perm)
                expected[j] = mask[i]

        env = BloodMahjongEnv()
        env._current_perm = perm
        actual = env._apply_augment_mask(mask)
        np.testing.assert_array_equal(actual, expected)

    def test_augment_dq_round_trip(self):
        from blood.env.augment import SUIT_PERMUTATIONS
        rng = np.random.default_rng(8)
        dq = rng.integers(0, 4, size=3).astype(np.float32)
        env = BloodMahjongEnv()
        for perm in SUIT_PERMUTATIONS[1:]:
            inv_perm = tuple(perm.index(i) for i in range(3))
            env._current_perm = perm
            aug = env._apply_augment_dq(dq)
            env._current_perm = inv_perm
            restored = env._apply_augment_dq(aug)
            np.testing.assert_array_equal(restored, dq)

    def test_augment_ow_round_trip(self):
        from blood.env.augment import SUIT_PERMUTATIONS
        rng = np.random.default_rng(9)
        ow = rng.integers(0, 3, size=81).astype(np.float32)
        env = BloodMahjongEnv()
        for perm in SUIT_PERMUTATIONS[1:]:
            inv_perm = tuple(perm.index(i) for i in range(3))
            env._current_perm = perm
            aug = env._apply_augment_ow(ow)
            env._current_perm = inv_perm
            restored = env._apply_augment_ow(aug)
            np.testing.assert_array_equal(restored, ow)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
