# Blood Arena (Human vs AI)

Sichuan Bloody Battle Mahjong - Human vs AI Interface.

## Prerequisites

- Python 3.10+
- `libblood` (compiled Rust module) in `PYTHONPATH` or installed.
- `mortal` (AI model code) in `PYTHONPATH`.
- **Trained model file** (e.g. `best.pth`). If missing, AI uses **random weights** and will be very weak.

## AI model (使用最新模型)

- Default path: `/data/mortal/mortal.pth`. If the file does not exist, the backend falls back to random init and logs a warning.
- To use a specific checkpoint, set the environment variable before starting:
  ```bash
  export MORTAL_MODEL=/path/to/your/latest.pth
  sh scripts/start_game.sh
  ```
- Or call `POST /start_game?ai_model=/path/to/latest.pth`, or send `{ "type": "start_game", "model": "/path/to/latest.pth" }` over the WebSocket. Check backend logs for `Loading AI model from:` to confirm which file was loaded.

## How to Run

1.  Navigate to the project root:
    ```bash
    cd /path/to/blood
    ```

2.  Run the Startup Script:
    ```bash
    sh scripts/start_game.sh
    ```

3.  Access the Game:
    Open your browser and navigate to:
    [http://localhost:8000/](http://localhost:8000/)

## Controls

- **Start Game**: Auto-starts on connection.
- **Discard**: Click a tile in your hand.
- **Actions**: Click buttons (Pon, Kan, Ron, Pass) when they appear.

## Troubleshooting

- **"Arena not available"**: Check if `libblood` is compiled and importable.
- **UI stuck at "Connecting"**: Refresh the page. Ensure backend is running.
