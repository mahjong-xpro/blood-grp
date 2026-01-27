#!/usr/bin/env python3
"""
Mahjong Log Replay Web Service
A web service for replaying Bloody Battle Mahjong game logs.
"""

import os
import sys
import gzip
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOG_VIEWER_DIR = Path(__file__).parent

# Log cache - stores full log content in memory
log_cache = {
    'logs': {},  # key: file path, value: {name, path, size, mtime, mtime_str, content, events, game_info}
    'log_list': [],  # List of log entries sorted by mtime (for display)
    'last_update': None,
    'lock': threading.Lock(),
}

# Default log directory (can be overridden by command line argument)
DEFAULT_LOG_DIR = None

@app.route('/')
def index():
    """Main page."""
    return render_template('replay.html')

def load_log_content(file_path):
    """Load and parse a log file, return log data."""
    try:
        file_path = Path(file_path)
        
        # Read log file
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                raw_log = f.read()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_log = f.read()
        
        # Parse events
        events = []
        for line in raw_log.strip().split('\n'):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                continue  # Skip invalid lines
        
        # Extract game info
        game_info = {
            'filename': file_path.name,
            'total_events': len(events),
            'events': events,
        }
        
        # Find start_game event
        for event in events:
            if event.get('type') == 'start_game':
                game_info['names'] = event.get('names', ['Player 0', 'Player 1', 'Player 2', 'Player 3'])
                game_info['seed'] = event.get('seed')
                break
        
        # Count P0 Agari
        p0_agari = 0
        for event in events:
            if event.get('type') == 'hora' and event.get('actor') == 0:
                p0_agari += 1
        print(f"DEBUG: Loaded {file_path.name}, P0 Wins: {p0_agari}")
        game_info['p0_agari_count'] = p0_agari
        
        return {
            'content': raw_log,
            'events': events,
            'game_info': game_info,
        }
    except Exception as e:
        print(f"Error loading log {file_path}: {e}")
        return None

def scan_and_cache_logs(log_dir, max_files=20):
    """Scan log directory and cache full log content in memory.
    Returns True if cache was updated, False if no changes detected.
    """
    if log_dir is None:
        return False
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return False
    
    new_logs = {}
    log_files = []
    has_new_or_updated = False
    
    # Scan for log files
    for ext in ['*.json', '*.json.gz']:
        for file_path in log_dir.rglob(ext):
            try:
                stat = file_path.stat()
                file_path_str = str(file_path)
                log_files.append({
                    'path': file_path_str,
                    'mtime': stat.st_mtime,
                })
            except (OSError, PermissionError):
                continue
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x['mtime'], reverse=True)
    
    # Check current cache state
    with log_cache['lock']:
        current_cache = log_cache['logs'].copy()
    
    # Load and cache top N files
    for log_file in log_files[:max_files]:
        file_path = log_file['path']
        try:
            stat = Path(file_path).stat()
            
            # Check if we already have this file cached and it hasn't changed
            cached = current_cache.get(file_path)
            if cached and cached.get('mtime') == stat.st_mtime:
                # File unchanged, keep existing cache
                new_logs[file_path] = cached
                continue
            
            # File is new or updated
            has_new_or_updated = True
            
            # Load new or updated file
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
        except (OSError, PermissionError) as e:
            print(f"Error accessing {file_path}: {e}")
            continue
    
    # If no new or updated files, check if we need to clean up old cache
    if not has_new_or_updated:
        with log_cache['lock']:
            # Check if we have too many cached files (including deleted ones)
            if len(log_cache['logs']) > max_files * 2:
                # Need to clean up, so we'll update
                has_new_or_updated = True
            else:
                # No changes detected, skip update
                return False
    
    # Update cache only if there are changes
    if has_new_or_updated:
        with log_cache['lock']:
            # Keep files that are no longer on disk but were cached
            # Only remove if we have too many cached files
            all_cached = {**log_cache['logs'], **new_logs}
            
            # If we have more than max_files * 2, remove oldest
            if len(all_cached) > max_files * 2:
                sorted_logs = sorted(all_cached.items(), key=lambda x: x[1].get('mtime', 0), reverse=True)
                all_cached = dict(sorted_logs[:max_files * 2])
            
            log_cache['logs'] = all_cached
            
            # Update sorted list for display
            log_cache['log_list'] = sorted(
                all_cached.values(),
                key=lambda x: x.get('mtime', 0),
                reverse=True
            )[:max_files]
            
            log_cache['last_update'] = datetime.now().isoformat()
        
        return True
    
    return False

def update_log_cache():
    """Update log cache in background."""
    global DEFAULT_LOG_DIR
    while True:
        try:
            if DEFAULT_LOG_DIR and DEFAULT_LOG_DIR.exists():
                updated = scan_and_cache_logs(DEFAULT_LOG_DIR, max_files=20)
                if updated:
                    with log_cache['lock']:
                        count = len(log_cache['log_list'])
                    print(f"[{datetime.now()}] Updated log cache: {count} files in memory")
                # else: no changes, skip update silently
        except Exception as e:
            print(f"[{datetime.now()}] Error updating log cache: {e}")
        
        time.sleep(10)  # Update every 10 seconds

@app.route('/api/logs', methods=['GET'])
def list_logs():
    """List available log files from cache."""
    custom_dir = request.args.get('dir')
    
    if custom_dir:
        # If custom directory is specified, scan it directly (without caching)
        if not os.path.exists(custom_dir):
            return jsonify({'error': 'Directory not found'}), 404
        
        log_files = []
        for ext in ['*.json', '*.json.gz']:
            for file_path in Path(custom_dir).rglob(ext):
                try:
                    stat = file_path.stat()
                    log_files.append({
                        'name': file_path.name,
                        'path': str(file_path),
                        'relative_path': str(file_path.relative_to(Path(custom_dir))),
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    })
                except (OSError, PermissionError):
                    continue
        
        log_files.sort(key=lambda x: x['mtime'], reverse=True)
        return jsonify({
            'logs': log_files[:100],
            'cached': False,
        })
    
    # Return cached logs (from memory)
    with log_cache['lock']:
        # Return display list (without full content)
        display_logs = []
        for log_entry in log_cache['log_list']:
            display_logs.append({
                'name': log_entry['name'],
                'path': log_entry['path'],
                'relative_path': log_entry.get('relative_path', ''),
                'size': log_entry['size'],
                'mtime': log_entry['mtime'],
                'mtime_str': log_entry['mtime_str'],
                'cached': True,  # Indicate this is cached in memory
                'cache_key': log_entry['path'],  # Use full path as cache key
                'p0_agari_count': log_entry.get('game_info', {}).get('p0_agari_count', 0),
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
    """Load and parse a log file. Try cache first, then file system."""
    try:
        # Decode URL-encoded path
        import urllib.parse
        log_path = urllib.parse.unquote(log_path)
        
        # First, try to get from memory cache
        with log_cache['lock']:
            # Try exact match first (most common case)
            cached_log = log_cache['logs'].get(log_path)
            
            # If not found, try multiple matching strategies
            if not cached_log:
                # Normalize the path for comparison
                log_path_normalized = str(Path(log_path).resolve()) if os.path.isabs(log_path) else log_path
                
                for cached_path, cached_entry in log_cache['logs'].items():
                    # Strategy 1: Exact match
                    if log_path == cached_path:
                        cached_log = cached_entry
                        break
                    
                    # Strategy 2: Match by filename
                    if log_path.endswith(cached_entry['name']) or cached_entry['name'] in log_path:
                        cached_log = cached_entry
                        break
                    
                    # Strategy 3: Match by relative path
                    rel_path = cached_entry.get('relative_path', '')
                    if rel_path and (log_path == rel_path or log_path.endswith(rel_path)):
                        cached_log = cached_entry
                        break
                    
                    # Strategy 4: Match by path components (handle different path separators)
                    cached_path_normalized = str(Path(cached_path).resolve()) if os.path.isabs(cached_path) else cached_path
                    if (log_path_normalized == cached_path_normalized or
                        log_path_normalized.endswith(cached_path_normalized) or
                        cached_path_normalized.endswith(log_path_normalized)):
                        cached_log = cached_entry
                        break
            
            if cached_log:
                # Return cached data
                print(f"Loading log from cache: {log_path} -> {cached_log['path']}")
                return jsonify(cached_log['game_info'])
        
        # If not in cache, try to load from file system (for manually specified files)
        # But first check if this is from the monitored directory - if so, file was deleted
        if DEFAULT_LOG_DIR:
            # Normalize paths for comparison
            try:
                log_path_normalized = str(Path(log_path).resolve())
                default_dir_normalized = str(DEFAULT_LOG_DIR.resolve())
                if log_path_normalized.startswith(default_dir_normalized):
                    return jsonify({
                        'error': f'Log file not found in cache: {log_path}. The file may have been deleted from disk. Please refresh the log list to see cached logs.'
                    }), 404
            except (OSError, ValueError):
                # If path resolution fails, try string comparison
                if str(log_path).startswith(str(DEFAULT_LOG_DIR)):
                    return jsonify({
                        'error': f'Log file not found in cache: {log_path}. The file may have been deleted from disk. Please refresh the log list to see cached logs.'
                    }), 404
        
        log_file = None
        
        # If it's an absolute path, use it directly
        if os.path.isabs(log_path):
            log_file = Path(log_path)
        else:
            # Try to find in default log directory first
            if DEFAULT_LOG_DIR:
                default_path = DEFAULT_LOG_DIR / log_path
                if default_path.exists():
                    log_file = default_path
                else:
                    # Try relative to project root
                    project_path = PROJECT_ROOT / log_path
                    if project_path.exists():
                        log_file = project_path
                    else:
                        # Try as absolute path from the path string
                        log_file = Path(log_path)
            else:
                # Try relative to project root
                project_path = PROJECT_ROOT / log_path
                if project_path.exists():
                    log_file = project_path
                else:
                    # Try as absolute path from the path string
                    log_file = Path(log_path)
        
        if not log_file or not log_file.exists():
            # If file doesn't exist and not in cache, return error
            return jsonify({
                'error': f'Log file not found: {log_path}. File may have been deleted. If this was from the monitored directory, please refresh the log list to load from cache.'
            }), 404
        
        # Load from file
        log_data = load_log_content(log_file)
        if not log_data:
            return jsonify({'error': 'Failed to load log file'}), 500
        
        return jsonify(log_data['game_info'])
    
    except Exception as e:
        import traceback
        print(f"Error in get_log: {e}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_log():
    """Upload a log file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save uploaded file
    upload_dir = LOG_VIEWER_DIR / 'uploads'
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    file.save(file_path)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'path': f'/api/log/uploads/{file.filename}'
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory(LOG_VIEWER_DIR / 'static', filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Mahjong Log Replay Web Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-dir', type=str, default='/data/mortal/train_play', 
                       help='Directory to scan for log files')
    args = parser.parse_args()
    
    # Update default log directory (global variable)
    DEFAULT_LOG_DIR = Path(args.log_dir)
    
    # Check if directory exists
    if not DEFAULT_LOG_DIR.exists():
        print(f"Warning: Log directory does not exist: {DEFAULT_LOG_DIR}")
        print("Log cache will be empty. You can still load logs manually.")
    else:
        # Start background thread to update log cache
        cache_thread = threading.Thread(target=update_log_cache, daemon=True)
        cache_thread.start()
        print(f"Started log cache updater thread (scanning: {DEFAULT_LOG_DIR})")
        
        # Initial cache update (load full content into memory)
        print(f"Performing initial cache update (loading full log content into memory)...")
        scan_and_cache_logs(DEFAULT_LOG_DIR, max_files=20)
        with log_cache['lock']:
            count = len(log_cache['log_list'])
            total_cached = len(log_cache['logs'])
        print(f"Initial cache: {count} files in display list, {total_cached} total cached in memory")
        print(f"Note: Logs are cached in memory. Even if files are deleted, cached logs remain available.")
        print(f"Cache will only update when new or modified files are detected.")
    
    print(f"Starting Mahjong Log Replay Web Service on http://{args.host}:{args.port}")
    print(f"Log directory: {DEFAULT_LOG_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug)
