"""WebSocket gateway for online Blood Mahjong play.

Provides a JSON-based protocol for clients to play against the AI.

Protocol:
    Client → Server:
        {"type": "new_game", "seed": 12345}
        {"type": "action", "action": 5}
        {"type": "get_state"}

    Server → Client:
        {"type": "state", "obs": [...], "mask": [...], "reward": 0.0,
         "terminated": false, "scores": [60000, 60000, 60000, 60000],
         "phase": "discard", "legal_actions": [0, 3, 5, 30]}
        {"type": "game_over", "scores": [...], "winner": 0}
        {"type": "error", "message": "..."}

Usage:
    python -m blood.serve.gateway --checkpoint path/to/model.pth --port 8765
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

ACTION_SPACE = 34


class GameSession:
    """Manages a single game session between a human player and AI opponents."""

    def __init__(self, model=None, use_rtpa=False, use_ismce=False):
        self._model = model
        self._use_rtpa = use_rtpa
        self._use_ismce = use_ismce
        self._env = None
        self._obs = None
        self._done = False

    def new_game(self, seed: int = 42):
        from blood.env.blood_env import BloodMahjongEnv
        self._env = BloodMahjongEnv()
        self._obs, _ = self._env.reset(seed=seed)
        self._done = False
        return self._get_state_response()

    def apply_action(self, action: int):
        if self._env is None:
            return {"type": "error", "message": "No active game. Send 'new_game' first."}
        if self._done:
            return {"type": "error", "message": "Game is over. Start a new game."}

        mask = self._obs["action_mask"]
        if mask[action] < 0.5:
            return {"type": "error", "message": f"Action {action} is not legal."}

        self._obs, reward, terminated, truncated, info = self._env.step(action)
        self._done = terminated or truncated

        resp = self._get_state_response()
        resp["reward"] = float(reward)

        if self._done:
            scores = [60000] * 4
            try:
                if self._env._env:
                    scores = list(self._env._env.get_scores())
            except Exception:
                pass
            resp["type"] = "game_over"
            resp["scores"] = scores
            winner = -1
            if isinstance(info, dict):
                winner = info.get("last_winner", info.get("player_won", -1))
                if isinstance(winner, bool):
                    winner = -1
            if winner == -1:
                max_score = max(scores)
                top_players = [i for i, s in enumerate(scores) if s == max_score]
                winner = top_players[0] if len(top_players) == 1 else -1
            resp["winner"] = winner
            resp["is_tie"] = winner == -1

        return resp

    def get_ai_suggestion(self) -> Optional[int]:
        """Get AI's recommended action for the current state."""
        if self._model is None or self._obs is None or self._done:
            return None

        obs_t = torch.as_tensor(self._obs["obs"], dtype=torch.float32).unsqueeze(0)
        mask = self._obs["action_mask"]

        with torch.no_grad():
            logits = self._model(obs_t).squeeze(0).numpy()

        logits[mask < 0.5] = -1e9
        return int(np.argmax(logits))

    def _get_state_response(self):
        mask = self._obs["action_mask"]
        legal = [int(i) for i in range(ACTION_SPACE) if mask[i] > 0.5]

        scores = [60000] * 4
        phase = "unknown"
        try:
            if self._env._env:
                scores = list(self._env._env.get_scores())
                phase = self._env._env.get_phase()
        except Exception:
            pass

        suggestion = self.get_ai_suggestion()

        return {
            "type": "state",
            "mask": mask.tolist(),
            "legal_actions": legal,
            "scores": scores,
            "phase": phase,
            "terminated": self._done,
            "ai_suggestion": suggestion,
        }


async def handle_connection(websocket, session_factory, auth_token=None):
    """Handle a single WebSocket connection."""
    if auth_token:
        try:
            first = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            msg = json.loads(first)
            if msg.get("type") != "auth" or msg.get("token") != auth_token:
                await websocket.send(json.dumps({"type": "error", "message": "Unauthorized"}))
                await websocket.close()
                return
            await websocket.send(json.dumps({"type": "auth_ok"}))
        except Exception:
            await websocket.close()
            return

    session = session_factory()
    log.info("Client connected: %s", websocket.remote_address)

    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({
                    "type": "error", "message": "Invalid JSON"
                }))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "new_game":
                seed = msg.get("seed", 42)
                response = session.new_game(seed=seed)

            elif msg_type == "action":
                action = msg.get("action", 30)
                response = session.apply_action(int(action))

            elif msg_type == "get_state":
                response = session._get_state_response()

            elif msg_type == "ai_action":
                suggestion = session.get_ai_suggestion()
                if suggestion is not None:
                    response = session.apply_action(suggestion)
                else:
                    response = {"type": "error", "message": "AI not available"}

            else:
                response = {"type": "error", "message": f"Unknown type: {msg_type}"}

            await websocket.send(json.dumps(response))

    except Exception as e:
        log.error("Connection error: %s", e)
    finally:
        log.info("Client disconnected")


async def run_server(host: str = "0.0.0.0", port: int = 8765, model=None, auth_token=None):
    try:
        import websockets
    except ImportError:
        log.error("websockets package not installed. Run: pip install websockets")
        return

    def session_factory():
        return GameSession(model=model)

    async with websockets.serve(
        lambda ws: handle_connection(ws, session_factory, auth_token=auth_token),
        host, port,
    ):
        auth_msg = " (token auth enabled)" if auth_token else " (no auth)"
        log.info("Blood Mahjong WebSocket server started on ws://%s:%d%s", host, port, auth_msg)
        await asyncio.Future()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Blood Mahjong WebSocket Gateway")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to SF2 model checkpoint for AI suggestions")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auth-token", type=str, default=None,
                        help="Optional token for client authentication")
    args = parser.parse_args()

    model = None
    if args.checkpoint:
        from blood.model.inference import PolicyModel
        model = PolicyModel.from_sf2_checkpoint(args.checkpoint)
        log.info("AI model loaded from %s", args.checkpoint)

    asyncio.run(run_server(host=args.host, port=args.port, model=model, auth_token=args.auth_token))


if __name__ == "__main__":
    main()
