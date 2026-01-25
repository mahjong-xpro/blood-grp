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

# Log cache
log_cache = {
    'logs': [],
    'last_update': None,
    'lock': threading.Lock(),
}

# Default log directory (can be overridden by command line argument)
DEFAULT_LOG_DIR = None

@app.route('/')
def index():
    """Main page."""
    return render_template('replay.html')

def scan_log_directory(log_dir, max_files=20):
    """Scan log directory and return latest log files."""
    if log_dir is None:
        return []
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return []
    
    log_files = []
    for ext in ['*.json', '*.json.gz']:
        for file_path in log_dir.rglob(ext):
            try:
                stat = file_path.stat()
                log_files.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'relative_path': str(file_path.relative_to(log_dir)),
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'mtime_str': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                })
            except (OSError, PermissionError):
                # Skip files that can't be accessed
                continue
    
    # Sort by modification time (newest first) and take top N
    log_files.sort(key=lambda x: x['mtime'], reverse=True)
    return log_files[:max_files]

def update_log_cache():
    """Update log cache in background."""
    global DEFAULT_LOG_DIR
    while True:
        try:
            if DEFAULT_LOG_DIR and DEFAULT_LOG_DIR.exists():
                logs = scan_log_directory(DEFAULT_LOG_DIR, max_files=20)
                with log_cache['lock']:
                    log_cache['logs'] = logs
                    log_cache['last_update'] = datetime.now().isoformat()
                print(f"[{datetime.now()}] Updated log cache: {len(logs)} files")
        except Exception as e:
            print(f"[{datetime.now()}] Error updating log cache: {e}")
        
        time.sleep(10)  # Update every 10 seconds

@app.route('/api/logs', methods=['GET'])
def list_logs():
    """List available log files from cache."""
    custom_dir = request.args.get('dir')
    
    if custom_dir:
        # If custom directory is specified, scan it directly
        if not os.path.exists(custom_dir):
            return jsonify({'error': 'Directory not found'}), 404
        
        log_files = scan_log_directory(custom_dir, max_files=100)
        return jsonify({
            'logs': log_files,
            'cached': False,
        })
    
    # Return cached logs
    with log_cache['lock']:
        return jsonify({
            'logs': log_cache['logs'],
            'last_update': log_cache['last_update'],
            'cached': True,
            'directory': str(DEFAULT_LOG_DIR) if DEFAULT_LOG_DIR else None,
        })

@app.route('/api/log/<path:log_path>')
def get_log(log_path):
    """Load and parse a log file."""
    try:
        # Try to find the log file
        log_file = None
        search_dirs = [
            PROJECT_ROOT,
            Path(log_path).parent if os.path.isabs(log_path) else PROJECT_ROOT,
        ]
        
        for base_dir in search_dirs:
            full_path = Path(base_dir) / log_path
            if full_path.exists():
                log_file = full_path
                break
        
        if not log_file or not log_file.exists():
            return jsonify({'error': f'Log file not found: {log_path}'}), 404
        
        # Read log file
        if log_file.suffix == '.gz':
            with gzip.open(log_file, 'rt', encoding='utf-8') as f:
                raw_log = f.read()
        else:
            with open(log_file, 'r', encoding='utf-8') as f:
                raw_log = f.read()
        
        # Parse events
        events = []
        for line in raw_log.strip().split('\n'):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON in log: {e}'}), 400
        
        # Extract game info
        game_info = {
            'filename': log_file.name,
            'total_events': len(events),
            'events': events,
        }
        
        # Find start_game event
        for event in events:
            if event.get('type') == 'start_game':
                game_info['names'] = event.get('names', ['Player 0', 'Player 1', 'Player 2', 'Player 3'])
                game_info['seed'] = event.get('seed')
                break
        
        return jsonify(game_info)
    
    except Exception as e:
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
        
        # Initial cache update
        print(f"Performing initial cache update...")
        logs = scan_log_directory(DEFAULT_LOG_DIR, max_files=20)
        with log_cache['lock']:
            log_cache['logs'] = logs
            log_cache['last_update'] = datetime.now().isoformat()
        print(f"Initial cache: {len(logs)} files")
    
    print(f"Starting Mahjong Log Replay Web Service on http://{args.host}:{args.port}")
    print(f"Log directory: {DEFAULT_LOG_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug)
