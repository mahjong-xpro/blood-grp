"""Persistent Elo rating tracker for Blood-v2 training.

Implements multi-player Elo using pairwise updates from 4-player mahjong games.
Ratings are persisted to disk and logged to TensorBoard.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Accumulated statistics for a single player/checkpoint."""

    elo: float = 1500.0
    games: int = 0
    wins: int = 0
    total_rank: float = 0.0  # sum of ranks for avg calculation
    total_score: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)

    @property
    def avg_rank(self) -> float:
        return self.total_rank / max(self.games, 1)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(self.games, 1)


class EloTracker:
    """Multi-player Elo rating system for mahjong.

    Uses pairwise Elo updates: in a 4-player game, each pair of players
    contributes an Elo update based on their relative ranking.

    K-factor adapts based on game count (higher K for new players).

    Ratings are persisted to a JSON file for continuity across restarts.
    Thread-safe: all mutations are guarded by a lock.
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        k_base: float = 32.0,
        k_new_player: float = 64.0,
        new_player_threshold: int = 30,
        base_rating: float = 1500.0,
    ):
        self.save_path = save_path
        self.k_base = k_base
        self.k_new_player = k_new_player
        self.new_player_threshold = new_player_threshold
        self.base_rating = base_rating
        self.players: dict[str, PlayerStats] = {}
        self._lock = threading.Lock()

        if save_path and os.path.exists(save_path):
            self.load()

    def _get_k(self, player_name: str) -> float:
        """Adaptive K-factor: higher for new players, lower for established ones."""
        stats = self.players.get(player_name)
        if stats is None or stats.games < self.new_player_threshold:
            return self.k_new_player
        return self.k_base

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Expected score of player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _ensure_player(self, name: str) -> PlayerStats:
        """Get or create player stats. Caller must hold self._lock."""
        if name not in self.players:
            self.players[name] = PlayerStats(elo=self.base_rating)
        return self.players[name]

    def update_from_game(
        self,
        player_names: list[str],
        ranks: list[float],
        scores: Optional[list[float]] = None,
    ) -> dict[str, float]:
        """Update Elo ratings from a single 4-player game result.

        Args:
            player_names: Names/identifiers for each player (length 4)
            ranks: Final ranking for each player (1.0 = first, 4.0 = last).
                   Supports fractional ranks for ties (e.g., 1.5 for tied 1st).
            scores: Optional final scores for stat tracking.

        Returns:
            Dict of {player_name: new_elo} for all players in the game.
        """
        n = len(player_names)
        assert n == len(ranks), f"Mismatched lengths: {n} names vs {len(ranks)} ranks"

        with self._lock:
            # Ensure all players exist
            stats = [self._ensure_player(name) for name in player_names]
            old_elos = [s.elo for s in stats]

            # Pairwise Elo updates
            elo_deltas = [0.0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    # Actual score: 1 if i ranked higher (lower rank number)
                    if ranks[i] < ranks[j]:
                        actual_i, actual_j = 1.0, 0.0
                    elif ranks[i] > ranks[j]:
                        actual_i, actual_j = 0.0, 1.0
                    else:
                        actual_i, actual_j = 0.5, 0.5

                    expected_i = self._expected_score(old_elos[i], old_elos[j])
                    expected_j = 1.0 - expected_i

                    # Average K of both players, scaled by 1/(n-1) for multi-player
                    k_i = self._get_k(player_names[i])
                    k_j = self._get_k(player_names[j])
                    k_avg = (k_i + k_j) / 2.0
                    scale = k_avg / (n - 1)

                    elo_deltas[i] += scale * (actual_i - expected_i)
                    elo_deltas[j] += scale * (actual_j - expected_j)

            # Apply updates and accumulate stats
            result = {}
            min_rank = min(ranks)
            for i, name in enumerate(player_names):
                stats[i].elo += elo_deltas[i]
                stats[i].games += 1
                stats[i].total_rank += ranks[i]
                # Count as win if player has the best (lowest) rank, including ties
                if ranks[i] <= min_rank:
                    stats[i].wins += 1
                if scores is not None and i < len(scores):
                    stats[i].total_score += scores[i]
                result[name] = stats[i].elo

        return result

    def get_rating(self, name: str) -> float:
        """Get current Elo rating for a player."""
        with self._lock:
            stats = self.players.get(name)
            return stats.elo if stats is not None else self.base_rating

    def get_stats(self, name: str) -> Optional[PlayerStats]:
        """Get full stats for a player, or None if unknown."""
        with self._lock:
            return self.players.get(name)

    def get_leaderboard(self, top_n: int = 20) -> list[tuple[str, PlayerStats]]:
        """Get top N players sorted by Elo rating.

        Args:
            top_n: Number of players to return. If <= 0, returns all players.
        """
        with self._lock:
            sorted_players = sorted(
                self.players.items(),
                key=lambda x: x[1].elo,
                reverse=True,
            )
            if top_n <= 0:
                return sorted_players  # Return all (Issue #51)
            return sorted_players[:top_n]

    def save(self) -> None:
        """Persist ratings to disk (atomic write via tmp + rename).

        Uses a unique tmp filename (with thread ID) to prevent concurrent
        save() calls from corrupting each other's tmp file.
        """
        if not self.save_path:
            return
        with self._lock:
            data = {
                "version": 1,
                "k_base": self.k_base,
                "base_rating": self.base_rating,
                "players": {
                    name: {
                        "elo": s.elo,
                        "games": s.games,
                        "wins": s.wins,
                        "total_rank": s.total_rank,
                        "total_score": s.total_score,
                    }
                    for name, s in self.players.items()
                },
            }
        # Write outside the lock to minimise hold time.
        # Use thread-unique tmp path to prevent concurrent writes from colliding.
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        tid = threading.get_ident()
        tmp_path = f"{self.save_path}.tmp.{tid}"
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self.save_path)
        log.debug("Saved Elo ratings for %d players to %s", len(data["players"]), self.save_path)

    def load(self) -> None:
        """Load ratings from disk."""
        if not self.save_path or not os.path.exists(self.save_path):
            return
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            with self._lock:
                for name, pdata in data.get("players", {}).items():
                    self.players[name] = PlayerStats(
                        elo=pdata["elo"],
                        games=pdata["games"],
                        wins=pdata["wins"],
                        total_rank=pdata.get("total_rank", 0.0),
                        total_score=pdata.get("total_score", 0.0),
                    )
            log.info("Loaded Elo ratings for %d players from %s", len(self.players), self.save_path)
        except Exception as e:
            log.warning("Failed to load Elo ratings: %s", e)
