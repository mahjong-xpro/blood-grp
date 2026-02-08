# Blood Arena (Human vs AI)

Sichuan Bloody Battle Mahjong - Human vs AI Interface.

## Prerequisites

- Python 3.10+
- `libblood` (compiled Rust module) in `PYTHONPATH` or installed.
- `mortal` (AI model code) in `PYTHONPATH`.
- Trained model file (e.g., `mortal.pth`).

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
