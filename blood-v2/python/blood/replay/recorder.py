"""Replay recorder: wraps RustMahjongEnv to save JSONL game logs."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class ReplayRecorder:
    """Saves completed games as JSONL replay files.

    Usage:
        recorder = ReplayRecorder(output_dir="replays/")
        # ... run game via arena ...
        recorder.save(rust_env, names=["Agent", "RuleBot", "RuleBot", "RuleBot"])
    """

    def __init__(self, output_dir: str, compress: bool = False, max_files: int = 500):
        self.output_dir = Path(output_dir)
        self.compress = compress
        self.max_files = max_files
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, rust_env, names: list | None = None) -> Path | None:
        """Serialize the completed game to a JSONL file. Returns the path or None on error."""
        if names is None:
            names = [f"Player{i}" for i in range(4)]
        try:
            header = rust_env.get_game_header_json(names)
            events_jsonl = rust_env.get_events_jsonl()
            final_scores = list(rust_env.get_final_scores())

            # Build lines: header + events (strip bare game_end) + enriched game_end
            lines = [header]
            for line in events_jsonl.splitlines():
                stripped = line.strip()
                if stripped and stripped != '{"type":"game_end"}':
                    lines.append(stripped)
            lines.append(json.dumps({"type": "game_end", "final_scores": final_scores}))

            content = "\n".join(lines) + "\n"
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            ext = ".json.gz" if self.compress else ".json"
            path = self.output_dir / f"game_{ts}{ext}"

            if self.compress:
                import gzip
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(content)
            else:
                path.write_text(content, encoding="utf-8")

            self._evict_old_files()
            return path
        except Exception as e:
            log.warning("ReplayRecorder.save failed: %s", e)
            return None

    def _evict_old_files(self):
        """Keep only the newest max_files replay files."""
        files = sorted(self.output_dir.glob("game_*.json*"), key=lambda p: p.stat().st_mtime)
        for old in files[: -self.max_files]:
            try:
                old.unlink()
            except OSError:
                pass
