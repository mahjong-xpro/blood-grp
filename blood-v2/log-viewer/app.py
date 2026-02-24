#!/usr/bin/env python3
"""
Blood V2 Mahjong Log Replay Web Service
Adapted from v1 log-viewer for blood-v2 event format.
"""

import os
import sys
import gzip
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory, abort
from flask_cors import CORS

app = Flask(__name__,
            template_folder='templates',
            static_folder=None)  # custom static route below
CORS(app)

LOG_VIEWER_DIR = Path(__file__).parent
# Fallback to v1 static assets (tile images + audio)
V1_STATIC = Path(__file__).parent.parent.parent / 'log-viewer' / 'static'

# Log cache
log_cache = {
    'logs': {},
    'log_list': [],
    'last_update': None,
    'lock': threading.Lock(),
}

DEFAULT_LOG_DIR = None


@app.route('/')
def index():
    return render_template('replay.html')


def load_log_content(file_path):
    """Load and parse a v2 JSONL log file."""
    try:
        file_path = Path(file_path)
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                raw_log = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_log = f.read()

        events = []
        for line in raw_log.strip().split('\n'):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        game_info = {
            'filename': file_path.name,
            'total_events': len(events),
            'events': events,
            'names': ['Player 0', 'Player 1', 'Player 2', 'Player 3'],
            'seed': None,
            'dealer': 0,
        }

        # Parse game_start header (v2 uses "game_start", v1 used "start_game")
        for event in events:
            if event.get('type') == 'game_start':
                game_info['names'] = event.get('names', game_info['names'])
                game_info['seed'] = event.get('seed')
                game_info['dealer'] = event.get('dealer', 0)
                break

        # Count wins per player (v2: tsumo + ron; v1: hora)
        win_counts = [0, 0, 0, 0]
        for event in events:
            if event.get('type') in ('tsumo', 'ron'):
                p = event.get('player')
                if p is not None and 0 <= p < 4:
                    win_counts[p] += 1
        game_info['player_agari_counts'] = win_counts

        # Hero detection
        hero_index = 0
        for i, name in enumerate(game_info['names']):
            if any(kw in name.lower() for kw in ('agent', 'neural', 'mortal', 'trainee')):
                hero_index = i
                break
        game_info['hero_index'] = hero_index
        game_info['hero_agari_count'] = win_counts[hero_index]

        return {'content': raw_log, 'events': events, 'game_info': game_info}
    except Exception as e:
        print(f"Error loading log {file_path}: {e}")
        return None


def scan_and_cache_logs(log_dir, max_files=20):
    if log_dir is None:
        return False
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return False

    new_logs = {}
    log_files = []
    has_new_or_updated = False

    for ext in ['*.json', '*.json.gz']:
        for file_path in log_dir.rglob(ext):
            try:
                stat = file_path.stat()
                log_files.append({'path': str(file_path), 'mtime': stat.st_mtime})
            except (OSError, PermissionError):
                continue

    log_files.sort(key=lambda x: x['mtime'], reverse=True)

    with log_cache['lock']:
        current_cache = log_cache['logs'].copy()

    for log_file in log_files[:max_files]:
        file_path = log_file['path']
        try:
            stat = Path(file_path).stat()
            cached = current_cache.get(file_path)
            if cached and cached.get('mtime') == stat.st_mtime:
                new_logs[file_path] = cached
                continue
            has_new_or_updated = True
            log_data = load_log_content(file_path)
            if log_data:
                new_logs[file_path] = {
                    'name': Path(file_path).name,
                    'path': file_path,
                    'relative_path': str(Path(file_path).relative_to(log_dir)),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'content': log_data['content'],
                    'events': log_data['events'],
                    'game_info': log_data['game_info'],
                }
        except (OSError, PermissionError):
            continue

    if not has_new_or_updated:
        with log_cache['lock']:
            if len(log_cache['logs']) > max_files * 2:
                has_new_or_updated = True
            else:
                return False

    if has_new_or_updated:
        with log_cache['lock']:
            all_cached = {**log_cache['logs'], **new_logs}
            if len(all_cached) > max_files * 2:
                sorted_logs = sorted(all_cached.items(), key=lambda x: x[1].get('mtime', 0), reverse=True)
                all_cached = dict(sorted_logs[:max_files * 2])
            log_cache['logs'] = all_cached
            log_cache['log_list'] = sorted(
                all_cached.values(), key=lambda x: x.get('mtime', 0), reverse=True
            )[:max_files]
            log_cache['last_update'] = datetime.now().isoformat()
        return True
    return False


def update_log_cache():
    global DEFAULT_LOG_DIR
    while True:
        try:
            if DEFAULT_LOG_DIR and DEFAULT_LOG_DIR.exists():
                updated = scan_and_cache_logs(DEFAULT_LOG_DIR, max_files=20)
                if updated:
                    with log_cache['lock']:
                        count = len(log_cache['log_list'])
                    print(f"[{datetime.now()}] Updated log cache: {count} files")
        except Exception as e:
            print(f"[{datetime.now()}] Error updating log cache: {e}")
        time.sleep(10)


@app.route('/api/logs', methods=['GET'])
def list_logs():
    with log_cache['lock']:
        display_logs = []
        for entry in log_cache['log_list']:
            gi = entry.get('game_info', {})
            win_counts = gi.get('player_agari_counts', [0, 0, 0, 0])
            hero_idx = gi.get('hero_index', 0)
            display_logs.append({
                'name': entry['name'],
                'path': entry['path'],
                'relative_path': entry.get('relative_path', ''),
                'size': entry['size'],
                'mtime': entry['mtime'],
                'mtime_str': entry['mtime_str'],
                'cached': True,
                'cache_key': entry['path'],
                'player_agari_counts': win_counts,
                'hero_index': hero_idx,
                'ordered_counts': [
                    {'count': win_counts[i], 'is_hero': i == hero_idx, 'idx': i}
                    for i in [hero_idx] + [x for x in range(4) if x != hero_idx]
                ],
            })
        return jsonify({
            'logs': display_logs,
            'last_update': log_cache['last_update'],
            'cached': True,
            'directory': str(DEFAULT_LOG_DIR) if DEFAULT_LOG_DIR else None,
            'total_cached': len(log_cache['logs']),
        })


@app.route('/api/log/<path:log_path>')
def get_log(log_path):
    try:
        import urllib.parse
        log_path = urllib.parse.unquote(log_path)

        with log_cache['lock']:
            cached_log = log_cache['logs'].get(log_path)
            if not cached_log:
                for cached_path, cached_entry in log_cache['logs'].items():
                    if log_path.endswith(cached_entry['name']) or cached_entry['name'] in log_path:
                        cached_log = cached_entry
                        break
            if cached_log:
                return jsonify(cached_log['game_info'])

        # Load from filesystem if not cached
        log_file = Path(log_path) if os.path.isabs(log_path) else (
            DEFAULT_LOG_DIR / log_path if DEFAULT_LOG_DIR else Path(log_path)
        )
        if not log_file or not log_file.exists():
            return jsonify({'error': f'Log file not found: {log_path}'}), 404

        log_data = load_log_content(log_file)
        if not log_data:
            return jsonify({'error': 'Failed to load log file'}), 500
        return jsonify(log_data['game_info'])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_log():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    upload_dir = LOG_VIEWER_DIR / 'uploads'
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    file.save(file_path)
    return jsonify({'message': 'File uploaded successfully', 'path': f'/api/log/uploads/{file.filename}'})


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files; fall back to v1 assets for tiles/audio."""
    local = LOG_VIEWER_DIR / 'static' / filename
    if local.exists():
        return send_from_directory(LOG_VIEWER_DIR / 'static', filename)
    if V1_STATIC.exists():
        v1_path = V1_STATIC / filename
        if v1_path.exists():
            return send_from_directory(V1_STATIC, filename)
    return abort(404)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Blood V2 Log Replay Web Service')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log-dir', type=str, default='replays',
                        help='Directory to scan for replay files')
    args = parser.parse_args()

    DEFAULT_LOG_DIR = Path(args.log_dir)

    if not DEFAULT_LOG_DIR.exists():
        print(f"Warning: Log directory does not exist: {DEFAULT_LOG_DIR}")
        print("Will create it when replays are saved.")
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    else:
        cache_thread = threading.Thread(target=update_log_cache, daemon=True)
        cache_thread.start()
        print(f"Scanning: {DEFAULT_LOG_DIR}")
        scan_and_cache_logs(DEFAULT_LOG_DIR, max_files=20)
        with log_cache['lock']:
            count = len(log_cache['log_list'])
        print(f"Initial cache: {count} files")

    print(f"Starting Blood V2 Replay on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
