"""Standardized evaluation protocol for Blood Mahjong agents.

Usage:
    python -m blood.eval.evaluate --checkpoint path/to/model.pth --num_games 2000

Supports:
    - RuleBot baseline evaluation
    - Neural opponent evaluation
    - RTPA and ISMCE enhanced evaluation
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

log = logging.getLogger(__name__)


class NeuralAgent:
    """Agent that uses a PolicyModel for action selection."""

    def __init__(self, model: PolicyModel, device: str = "cpu", temperature: float = 0.1):
        self.model = model
        self.device = device
        self.temperature = temperature
        self._rtpa = None
        self._ismce = None
        self._env_ref = None

    def enable_rtpa(self, attack_temp=0.8, defend_temp=1.5):
        from blood.eval.rtpa import RTPA
        self._rtpa = RTPA(attack_temp=attack_temp, defend_temp=defend_temp)

    def enable_ismce(self, num_worlds=64, rollout_depth=4):
        from blood.eval.ismce import ISMCESearcher
        self._ismce = ISMCESearcher(num_worlds=num_worlds, rollout_depth=rollout_depth)

    def set_env(self, env):
        """Allow the arena to pass the env reference for game state queries."""
        self._env_ref = env

    def _get_game_context(self):
        """Extract game state context for RTPA/ISMCE from the Rust env."""
        ctx = {
            "is_tenpai": False,
            "opponents_likely_tenpai": 0,
            "my_score": 60000,
            "avg_opponent_score": 60000.0,
            "wall_remaining": 50,
        }
        try:
            env = self._env_ref
            if env is None or env._env is None:
                return ctx
            rust = env._env
            scores = rust.get_scores()
            ctx["my_score"] = scores[0]
            ctx["avg_opponent_score"] = sum(scores[1:]) / 3.0

            if hasattr(rust, "get_wall_remaining"):
                ctx["wall_remaining"] = rust.get_wall_remaining()
            else:
                ctx["wall_remaining"] = 55

            if hasattr(rust, "get_is_tenpai"):
                ctx["is_tenpai"] = rust.get_is_tenpai(0)

            if hasattr(rust, "get_opponent_likely_tenpai"):
                ctx["opponents_likely_tenpai"] = rust.get_opponent_likely_tenpai(0)
            else:
                opp_tenpai = 0
                for pid in range(1, 4):
                    if hasattr(rust, "get_player_melds_count"):
                        if rust.get_player_melds_count(pid) >= 2:
                            opp_tenpai += 1
                ctx["opponents_likely_tenpai"] = opp_tenpai
        except Exception:
            pass
        return ctx

    @torch.no_grad()
    def __call__(self, obs_dict) -> int:
        obs = obs_dict["obs"]
        mask = obs_dict["action_mask"]

        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.model(obs_t).squeeze(0).numpy()

        if self._ismce is not None:
            ctx = self._get_game_context()
            hand_arr = None
            tiles_seen_arr = None
            melds_count = 0
            dq_int = -1
            try:
                env = self._env_ref
                if env is not None and env._env is not None:
                    rust = env._env
                    if hasattr(rust, "get_hand_counts"):
                        hand_arr = np.array(rust.get_hand_counts(0), dtype=np.uint8)
                    if hasattr(rust, "get_tiles_seen"):
                        tiles_seen_arr = np.array(rust.get_tiles_seen(0), dtype=np.uint8)
                    if hasattr(rust, "get_player_melds_count"):
                        melds_count = rust.get_player_melds_count(0)
                    if hasattr(rust, "get_ding_que"):
                        dq_int = rust.get_ding_que(0)
            except Exception:
                pass
            return self._ismce.select_action(
                logits, mask,
                hand=hand_arr,
                melds_count=melds_count,
                ding_que=dq_int,
                tiles_seen=tiles_seen_arr,
                wall_remaining=ctx["wall_remaining"],
                temperature=self.temperature,
            )

        if self._rtpa is not None:
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
            return int(np.random.choice(ACTION_SPACE, p=probs))

        logits[mask < 0.5] = -1e9
        logits /= max(self.temperature, 1e-8)
        probs = _softmax(logits)
        return int(np.random.choice(ACTION_SPACE, p=probs))


class RandomAgent:
    """Uniform random legal action agent (for baseline comparison)."""

    def __call__(self, obs_dict) -> int:
        mask = obs_dict["action_mask"]
        legal = np.where(mask > 0.5)[0]
        if len(legal) == 0:
            return 30  # Pass
        return int(np.random.choice(legal))


from blood.utils import softmax as _softmax


def run_evaluation(
    checkpoint_path: str = None,
    num_games: int = 2000,
    baseline: str = "rulebot",
    use_rtpa: bool = False,
    use_ismce: bool = False,
    temperature: float = 0.1,
    seed: int = 0,
    output_json: str = None,
) -> ArenaResult:
    """Run standardized evaluation and return results."""

    if checkpoint_path:
        model = PolicyModel.from_sf2_checkpoint(checkpoint_path)
        agent = NeuralAgent(model, temperature=temperature)
        if use_rtpa:
            agent.enable_rtpa()
        if use_ismce:
            agent.enable_ismce()
        agent_fn = agent
        agent_name = f"Neural({Path(checkpoint_path).stem})"
    else:
        agent_fn = RandomAgent()
        agent_name = "Random"

    log.info("Evaluating %s vs %s (%d games, seed=%d)", agent_name, baseline, num_games, seed)

    arena = Arena(BloodMahjongEnv, agent_fn, baseline_mode=baseline)
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
    )

    sys.exit(0 if result.win_rate > 0.0 else 1)


if __name__ == "__main__":
    main()
