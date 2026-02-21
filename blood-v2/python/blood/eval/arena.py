"""1v3 arena evaluation for Blood Mahjong agents."""

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ArenaResult:
    """Aggregated metrics from arena evaluation."""
    num_games: int = 0
    wins: int = 0
    total_rank: float = 0.0
    total_score: float = 0.0
    total_fan: float = 0.0
    win_fan_count: int = 0
    scores: list = field(default_factory=list)
    ranks: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.num_games, 1)

    @property
    def avg_rank(self) -> float:
        return self.total_rank / max(self.num_games, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.num_games, 1)

    @property
    def avg_fan(self) -> float:
        return self.total_fan / max(self.win_fan_count, 1)

    def confidence_interval(self, metric: str, confidence: float = 0.95, n_bootstrap: int = 10000):
        """Bootstrap confidence interval for a given metric."""
        rng = np.random.default_rng(42)
        data = getattr(self, f"{metric}s", None)
        if data is None or len(data) == 0:
            return (0.0, 0.0)
        data = np.array(data)
        n = len(data)
        means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            means.append(np.mean(sample))
        means = sorted(means)
        lo = means[int((1 - confidence) / 2 * n_bootstrap)]
        hi = means[int((1 + confidence) / 2 * n_bootstrap)]
        return (lo, hi)

    def summary(self) -> str:
        score_ci = self.confidence_interval("score")
        rank_ci = self.confidence_interval("rank")
        return (
            f"Games: {self.num_games}\n"
            f"Win rate: {self.win_rate:.3f}\n"
            f"Avg rank: {self.avg_rank:.3f} (95% CI: [{rank_ci[0]:.3f}, {rank_ci[1]:.3f}])\n"
            f"Avg score: {self.avg_score:.1f} (95% CI: [{score_ci[0]:.1f}, {score_ci[1]:.1f}])\n"
            f"Avg fan per win: {self.avg_fan:.2f}"
        )


class Arena:
    """Run 1v3 evaluation: agent vs 3 baselines."""

    def __init__(self, env_cls, agent_fn, baseline_mode="rulebot"):
        """
        env_cls: environment class
        agent_fn: callable(obs) -> action
        baseline_mode: opponent mode for baselines
        """
        self.env_cls = env_cls
        self.agent_fn = agent_fn
        self.baseline_mode = baseline_mode

    def evaluate(self, num_games: int = 1000, seed: int = 0) -> ArenaResult:
        """Run arena evaluation."""
        result = ArenaResult()
        rng = np.random.default_rng(seed)

        class _EvalCfg:
            suit_augment_prob = 0.0
            opponent_mode = self.baseline_mode

        for game_idx in range(num_games):
            game_seed = int(rng.integers(0, 2**32))
            env = self.env_cls(cfg=_EvalCfg())
            if hasattr(self.agent_fn, 'set_env'):
                self.agent_fn.set_env(env)
            obs, info = env.reset(seed=game_seed)
            done = False

            while not done:
                action = self.agent_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            try:
                scores = env._env.get_scores() if env._env else [60000] * 4
            except Exception:
                scores = [60000] * 4

            player_score = scores[0]
            rank = sum(1 for s in scores if s > player_score) + 1
            won = info.get("player_won", False) if isinstance(info, dict) else False

            result.num_games += 1
            result.total_score += player_score
            result.total_rank += rank
            result.scores.append(player_score)
            result.ranks.append(rank)
            if won:
                result.wins += 1
                fan = info.get("fan_count", 0) if isinstance(info, dict) else 0
                if not isinstance(fan, (int, float)):
                    fan = 0
                if fan > 0:
                    result.total_fan += fan
                    result.win_fan_count += 1

        return result
