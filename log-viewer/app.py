#!/usr/bin/env python3
"""
Mahjong Log Replay Web Service
A web service for replaying Bloody Battle Mahjong game logs.
"""

import os
import sys
import gzip
import json
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
LOG_VIEWER_DIR = Path(__file__).parent

@app.route('/')
def index():
    """Main page."""
    return render_template('replay.html')

@app.route('/api/logs', methods=['GET'])
def list_logs():
    """List available log files."""
    log_dir = request.args.get('dir', str(PROJECT_ROOT))
    if not os.path.exists(log_dir):
        return jsonify({'error': 'Directory not found'}), 404
    
    log_files = []
    for ext in ['*.json', '*.json.gz']:
        for file_path in Path(log_dir).rglob(ext):
            rel_path = str(file_path.relative_to(Path(log_dir)))
            log_files.append({
                'name': file_path.name,
                'path': rel_path,
                'size': file_path.stat().st_size,
            })
    
    return jsonify({'logs': sorted(log_files, key=lambda x: x['name'])})

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
    args = parser.parse_args()
    
    print(f"Starting Mahjong Log Replay Web Service on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
