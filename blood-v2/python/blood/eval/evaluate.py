"""Standardized evaluation protocol for Blood Mahjong agents.

Usage:
    python -m blood.eval.evaluate --checkpoint path/to/model.pth --num_games 2000

Supports:
    - RuleBot baseline evaluation
    - Neural opponent evaluation
    - RTPA and ISMCE enhanced evaluation (可协同工作)
    - Bootstrap confidence intervals
    - JSON result export
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from blood.env.blood_env import BloodMahjongEnv, OBS_SIZE, ACTION_SPACE
from blood.eval.arena import Arena, ArenaResult
from blood.model.inference import PolicyModel
from blood.utils import softmax as _softmax  # Fix R10-L5: moved from bottom of file
from blood.consts import (
    NUM_TILE_TYPES, NUM_STUDENT_CHANNELS,
    CH_HAND_BASE, CH_HAND_COUNT, CH_DING_QUE_BASE, CH_OPP_DING_QUE_BASE,
    CH_TILES_REMAINING, CH_SELF_MELDS, CH_OPP_MELD_BASE, CH_OPP_KAWA_OVERVIEW_BASE,
    TILES_PER_SUIT,
)

log = logging.getLogger(__name__)

# COPIES_PER_TILE（每张牌的副本数）
_COPIES_PER_TILE = 4
# MAX_MELDS（最大副露数）
_MAX_MELDS = 4
# 定缺花色数
_CH_DING_QUE_COUNT = 3  # Man=0, Pin=1, Sou=2


def _extract_ismce_state(obs: np.ndarray) -> dict:
    """从 470×27 观测张量中提取 ISMCE 所需的游戏状态信息。

    解析已知通道偏移量（来自 crates/engine/src/obs/student.rs）：
    - 通道 0-3 (Section 1): 手牌 one-hot 编码 → hand[t] = sum(ch0..ch3)
    - 通道 18-20 (Section 3): 定缺花色 → ding_que suit index
    - 通道 329 (Section 9): 每张牌剩余数 → tiles_seen = 4 - round(val * 4)
    - 通道 331 (Section 9): 自家副露数 → melds_count = round(val * 4)

    Returns:
        dict 包含 hand, tiles_seen, melds_count, ding_que；
        提取失败时返回 None 值，调用方应做 fallback 处理。
    """
    result = {"hand": None, "tiles_seen": None, "melds_count": 0, "ding_que": -1}

    try:
        # 将一维观测重塑为 (NUM_STUDENT_CHANNELS, NUM_TILE_TYPES)
        if obs.size < NUM_STUDENT_CHANNELS * NUM_TILE_TYPES:
            log.debug("观测张量尺寸不足，无法提取 ISMCE 状态")
            return result
        # Fix R10-M11: slice to student channels to handle oracle obs (525×27)
        obs_2d = obs[:NUM_STUDENT_CHANNELS * NUM_TILE_TYPES].reshape(NUM_STUDENT_CHANNELS, NUM_TILE_TYPES)

        # ── 提取手牌 ──
        # 通道 0-3 是 one-hot 编码：ch_k[t] = 1.0 if hand[t] > k
        # 因此 hand[t] = sum(ch0[t], ch1[t], ch2[t], ch3[t])
        hand = np.zeros(NUM_TILE_TYPES, dtype=np.uint8)
        for k in range(CH_HAND_COUNT):
            hand += (obs_2d[CH_HAND_BASE + k] > 0.5).astype(np.uint8)
        result["hand"] = hand

        # ── 提取可见牌数 ──
        # 通道 329 编码: remaining[t] / 4.0，其中 remaining = 4 - tiles_seen[t]
        # 因此 tiles_seen[t] = 4 - round(obs_2d[329][t] * 4)
        remaining_ratio = obs_2d[CH_TILES_REMAINING]
        tiles_seen = np.clip(
            _COPIES_PER_TILE - np.round(remaining_ratio * _COPIES_PER_TILE).astype(np.int32),
            0, _COPIES_PER_TILE,
        ).astype(np.uint8)
        result["tiles_seen"] = tiles_seen

        # ── 提取副露数 ──
        # 通道 331 编码: melds.len() / MAX_MELDS
        melds_ratio = float(obs_2d[CH_SELF_MELDS].mean())
        result["melds_count"] = max(0, min(int(round(melds_ratio * _MAX_MELDS)), _MAX_MELDS))

        # ── 提取定缺花色 ──
        # 通道 18-20 分别对应 万(0)/筒(1)/条(2)，花色通道内对应牌位置为 1.0
        ding_que = -1
        for suit_idx in range(_CH_DING_QUE_COUNT):
            ch_val = obs_2d[CH_DING_QUE_BASE + suit_idx]
            # 检查该花色对应的 9 张牌位置是否有非零值
            suit_start = suit_idx * TILES_PER_SUIT
            suit_end = suit_start + TILES_PER_SUIT
            if np.any(ch_val[suit_start:suit_end] > 0.5):
                ding_que = suit_idx
                break
        result["ding_que"] = ding_que

    except Exception as e:
        log.debug("提取 ISMCE 状态失败: %s", e)

    return result


def _extract_opponent_state(obs: np.ndarray) -> dict:
    """从观测张量中提取对手状态信息，用于 ISMCE 全量评估。

    解析通道：
    - 通道 23-31 (Section 3): 对手定缺花色 (3 opponents × 3 suits)
    - 通道 333-335 (Section 9): 对手副露数
    - Section 7 (ch 272+): 对手打牌概览 (3 × 4 one-hot) → 推算打牌数

    Returns:
        dict with opponent_ding_que, opponent_meld_counts,
        opponent_discard_counts, opponent_discards;
        提取失败时返回 None，调用方应做 fallback 处理。
    """
    try:
        if obs.size < NUM_STUDENT_CHANNELS * NUM_TILE_TYPES:
            return None
        # Fix R10-M11: slice to student channels to handle oracle obs (525×27)
        obs_2d = obs[:NUM_STUDENT_CHANNELS * NUM_TILE_TYPES].reshape(NUM_STUDENT_CHANNELS, NUM_TILE_TYPES)

        # ── 对手定缺花色 ──
        opponent_ding_que = []
        for opp_idx in range(3):
            dq = -1
            base_ch = CH_OPP_DING_QUE_BASE + opp_idx * 3
            for suit_idx in range(3):
                ch_val = obs_2d[base_ch + suit_idx]
                suit_start = suit_idx * TILES_PER_SUIT
                suit_end = suit_start + TILES_PER_SUIT
                if np.any(ch_val[suit_start:suit_end] > 0.5):
                    dq = suit_idx
                    break
            opponent_ding_que.append(dq)

        # ── 对手副露数 ──
        opponent_meld_counts = []
        for opp_idx in range(3):
            meld_ratio = float(obs_2d[CH_OPP_MELD_BASE + opp_idx].mean())
            mc = max(0, min(int(round(meld_ratio * _MAX_MELDS)), _MAX_MELDS))
            opponent_meld_counts.append(mc)

        # ── 对手打牌数和最近打牌 ──
        # Section 7 kawa overview: 3 opponents × 4 one-hot channels
        # counts[t] = sum of one-hot layers > 0.5 → how many times opp discarded tile t
        opponent_discard_counts = []
        opponent_discards = []
        for opp_idx in range(3):
            base_ch = CH_OPP_KAWA_OVERVIEW_BASE + opp_idx * 4
            counts = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
            for k in range(4):
                counts += (obs_2d[base_ch + k] > 0.5).astype(np.int32)
            total_discards = int(counts.sum())
            opponent_discard_counts.append(total_discards)
            # Reconstruct discard list (tiles with counts > 0, repeated)
            disc = []
            for t in range(NUM_TILE_TYPES):
                for _ in range(counts[t]):
                    disc.append(t)
            opponent_discards.append(disc)

        return {
            "opponent_ding_que": opponent_ding_que,
            "opponent_meld_counts": opponent_meld_counts,
            "opponent_discard_counts": opponent_discard_counts,
            "opponent_discards": opponent_discards,
        }

    except Exception as e:
        log.debug("提取对手状态失败: %s", e)
        return None


class NeuralAgent:
    """Agent that uses a PolicyModel for action selection."""

    def __init__(self, model: PolicyModel, device: str = "cpu", temperature: float = 0.1,
                 rng: np.random.Generator | None = None):
        self.model = model
        self.device = device
        self.temperature = temperature
        self._rng = rng if rng is not None else np.random.default_rng()
        self._rtpa = None
        self._ismce = None
        self._env_ref = None
        self._hidden_state = None  # LSTM hidden state across turns
        self._memory_buffer = None  # TurnAttention memory buffer across turns
        self._last_obs = None
        self._agent_seat = 0  # Track agent seat for correct score extraction

    def enable_rtpa(self, attack_temp=0.8, defend_temp=1.5):
        from blood.eval.rtpa import RTPA
        self._rtpa = RTPA(attack_temp=attack_temp, defend_temp=defend_temp)

    def enable_ismce(self, num_worlds=64, rollout_depth=4):
        from blood.eval.ismce import ISMCESearcher
        self._ismce = ISMCESearcher(num_worlds=num_worlds, rollout_depth=rollout_depth)

    def set_env(self, env):
        """Allow the arena to pass the env reference for game state queries."""
        self._env_ref = env
        self._hidden_state = None  # reset LSTM state at episode boundary
        self._memory_buffer = None  # reset TurnAttention memory at episode boundary

    def set_agent_seat(self, seat: int):
        """Set the agent's seat index for correct score extraction."""
        self._agent_seat = seat

    def _get_game_context(self):
        """从环境公共 API 提取 RTPA/ISMCE 所需的游戏状态上下文。"""
        ctx = {
            "is_tenpai": False,
            "opponents_likely_tenpai": 0,
            "my_score": 100000,
            "avg_opponent_score": 100000.0,
            "wall_remaining": 50,
        }
        try:
            env = self._env_ref
            if env is None or not env.has_engine:
                return ctx
            scores = env.get_scores()
            seat = self._agent_seat
            ctx["my_score"] = scores[seat]
            opp_scores = [scores[i] for i in range(len(scores)) if i != seat]
            ctx["avg_opponent_score"] = sum(opp_scores) / max(len(opp_scores), 1)

            # 使用 GameStateTracker 从观测张量解析听牌/牌墙信息
            if self._last_obs is not None:
                from blood.eval.rtpa import GameStateTracker
                if not hasattr(self, "_tracker"):
                    self._tracker = GameStateTracker()
                self._tracker.update_from_obs(
                    self._last_obs, scores=scores, agent_seat=seat,
                )
                ctx["is_tenpai"] = self._tracker.my_tenpai
                ctx["opponents_likely_tenpai"] = self._tracker.opponents_tenpai_count
                ctx["wall_remaining"] = self._tracker.wall_remaining
        except Exception:
            pass
        return ctx

    @torch.no_grad()
    def __call__(self, obs_dict) -> int:
        obs = obs_dict["obs"]
        mask = obs_dict["action_mask"]
        self._last_obs = obs  # 缓存用于 _get_game_context

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        # PolicyModel.forward() returns (logits, new_hidden_state, new_memory_buffer)
        logits_t, self._hidden_state, self._memory_buffer = self.model(
            obs_t, self._hidden_state, memory_buffer=self._memory_buffer,
        )
        logits = logits_t.squeeze(0).numpy()

        # ── 第一步：RTPA 计算自适应温度 ──
        # 无论是否启用 ISMCE，只要启用了 RTPA 就先计算动态温度
        temperature = self.temperature  # 默认使用固定温度
        ctx = None
        if self._rtpa is not None:
            ctx = self._get_game_context()
            temperature = self._rtpa.compute_temperature(
                is_tenpai=ctx["is_tenpai"],
                opponents_likely_tenpai=ctx["opponents_likely_tenpai"],
                my_score=ctx["my_score"],
                avg_opponent_score=ctx["avg_opponent_score"],
                wall_remaining=ctx["wall_remaining"],
            )

        # ── 第二步：ISMCE 搜索（使用 RTPA 温度 + 完整游戏状态）──
        if self._ismce is not None:
            if ctx is None:
                ctx = self._get_game_context()

            # 从观测张量提取 ISMCE 所需的游戏状态
            ismce_state = _extract_ismce_state(obs)

            # 提取对手状态用于全量评估（约束采样+防守）
            opp_state = _extract_opponent_state(obs)

            try:
                kwargs = dict(
                    hand=ismce_state["hand"],
                    tiles_seen=ismce_state["tiles_seen"],
                    melds_count=ismce_state["melds_count"],
                    ding_que=ismce_state["ding_que"],
                    wall_remaining=ctx["wall_remaining"],
                    temperature=temperature,  # 使用 RTPA 自适应温度（或默认温度）
                )
                # Pass opponent state if extraction succeeded
                if opp_state is not None:
                    kwargs.update(opp_state)

                action = self._ismce.select_action(logits, mask, **kwargs)
                return action
            except Exception as e:
                log.debug("ISMCE select_action 异常，回退到策略网络: %s", e)
                # ISMCE 失败时 fallback 到下面的逻辑

        # ── 第三步：纯 RTPA 路径（无 ISMCE 时）──
        if self._rtpa is not None:
            if ctx is None:
                ctx = self._get_game_context()
            logits = self._rtpa.adapt_logits(
                logits, mask,
                is_tenpai=ctx["is_tenpai"],
                opponents_likely_tenpai=ctx["opponents_likely_tenpai"],
                my_score=ctx["my_score"],
                avg_opponent_score=ctx["avg_opponent_score"],
                wall_remaining=ctx["wall_remaining"],
            )
            probs = _softmax(logits)
            return int(self._rng.choice(ACTION_SPACE, p=probs))

        # ── 第四步：纯策略网络（无 RTPA 无 ISMCE）──
        logits[mask < 0.5] = -1e38  # safe for both float32 and float16
        logits /= max(self.temperature, 1e-8)
        probs = _softmax(logits)
        return int(self._rng.choice(ACTION_SPACE, p=probs))


class RandomAgent:
    """Uniform random legal action agent (for baseline comparison)."""

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, obs_dict) -> int:
        mask = obs_dict["action_mask"]
        legal = np.where(mask > 0.5)[0]
        if len(legal) == 0:
            return 30  # Pass
        return int(self._rng.choice(legal))



def run_evaluation(
    checkpoint_path: str = None,
    num_games: int = 2000,
    baseline: str = "rulebot",
    use_rtpa: bool = False,
    use_ismce: bool = False,
    temperature: float = 0.1,
    seed: int = 0,
    output_json: str = None,
    save_replays: str = None,
) -> ArenaResult:
    """Run standardized evaluation and return results."""

    if checkpoint_path:
        model = PolicyModel.from_sf2_checkpoint(checkpoint_path)
        agent_rng = np.random.default_rng(seed)
        agent = NeuralAgent(model, temperature=temperature, rng=agent_rng)
        if use_rtpa:
            agent.enable_rtpa()
        if use_ismce:
            agent.enable_ismce()
        agent_fn = agent
        agent_name = f"Neural({Path(checkpoint_path).stem})"
    else:
        agent_fn = RandomAgent(rng=np.random.default_rng(seed))
        agent_name = "Random"

    log.info("Evaluating %s vs %s (%d games, seed=%d)", agent_name, baseline, num_games, seed)

    recorder = None
    if save_replays:
        from blood.replay.recorder import ReplayRecorder
        recorder = ReplayRecorder(save_replays, compress=False, max_files=500)
        log.info("Replay recording enabled → %s", save_replays)

    arena = Arena(BloodMahjongEnv, agent_fn, baseline_mode=baseline, recorder=recorder)
    t0 = time.time()
    result = arena.evaluate(num_games=num_games, seed=seed)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Agent: {agent_name}")
    print(f"Baseline: {baseline}")
    print(f"RTPA: {'ON' if use_rtpa else 'OFF'}  |  ISMCE: {'ON' if use_ismce else 'OFF'}")
    print(f"{'='*50}")
    print(result.summary())
    print(f"Time: {elapsed:.1f}s ({num_games/elapsed:.1f} games/s)")
    print(f"{'='*50}\n")

    if output_json:
        data = {
            "agent": agent_name,
            "baseline": baseline,
            "num_games": result.num_games,
            "win_rate": result.win_rate,
            "avg_rank": result.avg_rank,
            "avg_score": result.avg_score,
            "avg_fan": result.avg_fan,
            "rtpa": use_rtpa,
            "ismce": use_ismce,
            "elapsed_seconds": elapsed,
            "score_ci_95": list(result.confidence_interval("score")),
            "rank_ci_95": list(result.confidence_interval("rank")),
        }
        Path(output_json).write_text(json.dumps(data, indent=2))
        log.info("Results saved to %s", output_json)

    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Blood Mahjong Agent Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SF2 model checkpoint (.pth)")
    parser.add_argument("--num_games", type=int, default=2000)
    parser.add_argument("--baseline", type=str, default="rulebot",
                        choices=["rulebot", "random"])
    parser.add_argument("--rtpa", action="store_true", help="Enable RTPA")
    parser.add_argument("--ismce", action="store_true", help="Enable ISMCE")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--save_replays", type=str, default=None,
                        help="Directory to save JSONL replay files (one per game)")
    args = parser.parse_args()

    result = run_evaluation(
        checkpoint_path=args.checkpoint,
        num_games=args.num_games,
        baseline=args.baseline,
        use_rtpa=args.rtpa,
        use_ismce=args.ismce,
        temperature=args.temperature,
        seed=args.seed,
        output_json=args.output,
        save_replays=args.save_replays,
    )

    sys.exit(0 if result.win_rate > 0.0 else 1)


if __name__ == "__main__":
    main()
