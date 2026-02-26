"""1v3 竞技场评估：Blood 麻将 agent 对战评估。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from blood.eval.elo import EloTracker

log = logging.getLogger(__name__)


@dataclass
class ArenaResult:
    """竞技场评估的聚合指标。"""
    num_games: int = 0
    wins: int = 0
    total_rank: float = 0.0
    total_score: float = 0.0
    total_fan: float = 0.0
    win_fan_count: int = 0
    scores: list = field(default_factory=list)
    ranks: list = field(default_factory=list)
    agent_elo: float = 1500.0
    baseline_elo: float = 1500.0

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
        """给定指标的 Bootstrap 置信区间。"""
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
        lo_idx = min(int((1 - confidence) / 2 * n_bootstrap), n_bootstrap - 1)
        hi_idx = min(int((1 + confidence) / 2 * n_bootstrap), n_bootstrap - 1)
        lo = means[lo_idx]
        hi = means[hi_idx]
        return (lo, hi)

    def summary(self) -> str:
        score_ci = self.confidence_interval("score")
        rank_ci = self.confidence_interval("rank")
        lines = [
            f"Games: {self.num_games}",
            f"Win rate: {self.win_rate:.3f}",
            f"Avg rank: {self.avg_rank:.3f} (95% CI: [{rank_ci[0]:.3f}, {rank_ci[1]:.3f}])",
            f"Avg score: {self.avg_score:.1f} (95% CI: [{score_ci[0]:.1f}, {score_ci[1]:.1f}])",
            f"Avg fan per win: {self.avg_fan:.2f}",
            f"Agent Elo: {self.agent_elo:.1f}",
            f"Baseline Elo: {self.baseline_elo:.1f}",
        ]
        return "\n".join(lines)


def _compute_rank_with_ties(scores: list, player_idx: int) -> float:
    """计算考虑同分平均排名的名次。

    同分时使用平均排名而非最低排名，避免系统性偏差。
    例如：分数 [100, 100, 80, 60]，前两名同分，排名为 1.5, 1.5, 3, 4
    """
    player_score = scores[player_idx]
    n = len(scores)

    # 找出所有与 player_score 相同的玩家
    sorted_scores = sorted(scores, reverse=True)

    # 计算同分组的平均排名
    # 先找到该分数在排序后的起始和结束位置
    first_pos = None
    last_pos = None
    for pos, s in enumerate(sorted_scores):
        if s == player_score:
            if first_pos is None:
                first_pos = pos
            last_pos = pos

    # 平均排名 = (起始排名 + 结束排名) / 2，排名从1开始
    avg_rank = ((first_pos + 1) + (last_pos + 1)) / 2.0
    return avg_rank


class Arena:
    """运行 1v3 评估：agent 对战 3 个基线。"""

    def __init__(
        self,
        env_cls,
        agent_fn,
        baseline_mode="rulebot",
        recorder=None,
        elo_tracker: Optional[EloTracker] = None,
    ):
        """
        env_cls: 环境类
        agent_fn: callable(obs) -> action
        baseline_mode: 基线对手模式
        recorder: 可选的 ReplayRecorder 实例
        elo_tracker: 可选的 EloTracker 实例，用于跟踪 Elo 评分
        """
        self.env_cls = env_cls
        self.agent_fn = agent_fn
        self.baseline_mode = baseline_mode
        self.recorder = recorder
        self.elo_tracker = elo_tracker

    def evaluate(self, num_games: int = 1000, seed: int = 0,
                 names: list | None = None,
                 agent_name: str = "current_policy") -> ArenaResult:
        """运行竞技场评估。

        每局游戏随机选择 agent 的座位（0-3），消除庄家位偏差。
        """
        result = ArenaResult()
        rng = np.random.default_rng(seed)
        if names is None:
            names = ["Agent", "RuleBot", "RuleBot", "RuleBot"]

        class _EvalCfg:
            suit_augment_prob = 0.0
            opponent_mode = self.baseline_mode

        for game_idx in range(num_games):
            game_seed = int(rng.integers(0, 2**32))

            # 随机选择 agent 座位（0-3），消除固定庄家位的系统性偏差
            agent_seat = int(rng.integers(0, 4))

            env = self.env_cls(cfg=_EvalCfg())

            # 如果环境支持设置 agent 座位，则传入
            if hasattr(env, 'set_agent_seat'):
                env.set_agent_seat(agent_seat)

            if hasattr(self.agent_fn, 'set_env'):
                self.agent_fn.set_env(env)
            if hasattr(self.agent_fn, 'set_agent_seat'):
                self.agent_fn.set_agent_seat(agent_seat)
            obs, info = env.reset(seed=game_seed)
            done = False

            while not done:
                action = self.agent_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            try:
                scores = env.get_scores()
            except Exception:
                scores = [100000] * 4

            # 构建带座位轮换的名称列表用于录像和 Elo
            # Fix R10-H4: use distinct names for each RuleBot seat to prevent
            # 3x stat inflation and noisy self-play Elo updates on the same object.
            rotated_names = [f"RuleBot_{i}" for i in range(4)]
            rotated_names[agent_seat] = agent_name  # Use agent_name consistently

            if self.recorder is not None and env.has_engine:
                try:
                    self.recorder.save(env, names=rotated_names)
                except Exception as e:
                    log.warning("Recorder save failed for game %d: %s", game_idx, e)

            # 从 agent 实际座位提取分数
            player_score = scores[agent_seat]

            # 使用同分平均排名，避免同分时的排名偏差
            rank = _compute_rank_with_ties(scores, agent_seat)

            won = False
            if isinstance(info, dict):
                # Prefer the per-seat winners list from the Rust env (Issue #44)
                winners = info.get("winners", None)
                if winners is not None and len(winners) > 0:
                    won = agent_seat in winners
                else:
                    # Backward compatibility: fall back to older info fields
                    winner_seat = info.get("winner_seat", None)
                    if winner_seat is not None:
                        won = (winner_seat == agent_seat)
                    elif agent_seat == 0:
                        won = info.get("player_won", False)
                    else:
                        # Last resort: score-based heuristic
                        won = all(player_score >= s for s in scores) and player_score > min(scores)

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

            # Update Elo ratings for this game
            if self.elo_tracker is not None:
                # rotated_names already has agent_name at agent_seat
                all_ranks = [_compute_rank_with_ties(scores, i) for i in range(4)]
                self.elo_tracker.update_from_game(
                    player_names=rotated_names,
                    ranks=all_ranks,
                    scores=scores,
                )

        # Finalize Elo stats on the result
        if self.elo_tracker is not None and result.num_games > 0:
            result.agent_elo = self.elo_tracker.get_rating(agent_name)
            # Average baseline Elo across unique baseline names
            baseline_names = {n for n in rotated_names if n != agent_name}
            if baseline_names:
                result.baseline_elo = sum(
                    self.elo_tracker.get_rating(b) for b in baseline_names
                ) / len(baseline_names)
            self.elo_tracker.save()

        return result
