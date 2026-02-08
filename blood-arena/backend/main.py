import sys
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import queue
import asyncio

# Setup paths to import sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../blood-arena/backend
arena_root = os.path.dirname(current_dir)                 # .../blood-arena（前端所在目录）
parent_dir = os.path.dirname(arena_root)                  # Mahjong/blood
sys.path.append(parent_dir)

# Set MORTAL_CFG for mortal module
os.environ['MORTAL_CFG'] = os.path.join(parent_dir, 'mortal', 'config.toml')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Import GameManager
from backend.game_manager import GameManager

app = FastAPI()

# 静态资源与首页：使用绝对路径，不依赖进程 CWD
_frontend = os.path.join(arena_root, "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(_frontend, "static")), name="static")
app.mount("/js", StaticFiles(directory=os.path.join(_frontend, "js")), name="js")
app.mount("/css", StaticFiles(directory=os.path.join(_frontend, "css")), name="css")

# Initialize Game Manager
game_manager = GameManager()

@app.get("/")
async def get():
    index_path = os.path.join(_frontend, "index.html")
    if not os.path.isfile(index_path):
        logging.error("index.html not found at %s", index_path)
    return FileResponse(index_path)

@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Reconnection: send latest state (state_update or game_over) so client can restore UI
    if 'latest' in game_manager.shared_state:
        await websocket.send_json(game_manager.shared_state['latest'])

    model_path = os.path.join(parent_dir, "mortal", "models", "best.pth")

    # Listener loop for Queue -> WebSocket
    async def sender():
        while True:
            # Non-blocking get from queue? No, we need async wait
            # But queue is thread-safe, not asyncio-aware.
            # We can use run_in_executor to wait for queue
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, game_manager.state_queue.get
                )
                if msg.get("type") == "_thread_finished":
                    break
                await websocket.send_json(msg)
            except Exception as e:
                logging.error(f"Sender error: {e}")
                break

    # Listener loop for WebSocket -> Queue
    async def receiver():
        try:
            while True:
                data = await websocket.receive_json()
                if data.get("type") == "start_game":
                    # Start game thread only when user clicks "开始对局" (single game per start)
                    game_manager.start_game_thread(model_path)
                    continue
                game_manager.action_queue.put(data)
        except WebSocketDisconnect:
            logging.info("Client disconnected")
        except Exception as e:
            logging.error(f"Receiver error: {e}")

    # Run both
    await asyncio.gather(sender(), receiver())
