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

@app.post("/start_game")
async def start_game(ai_model: str = None):
    # Determine model path
    model_path = ai_model
    if not model_path:
        model_path = os.environ.get('MORTAL_MODEL', '/data/mortal/mortal.pth')
    
    # Start thread
    try:
        game_manager.start_game_thread(model_path)
        return {"status": "started", "model": model_path}
    except Exception as e:
        logging.error(f"Failed to start game: {e}")
        return {"status": "error", "message": str(e)}

# Startup/Shutdown events to manage broadcaster
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_loop())

async def broadcast_loop():
    logging.info("Global broadcast loop started")
    while True:
        try:
            msg = await asyncio.get_event_loop().run_in_executor(
                None, game_manager.state_queue.get
            )
            if msg.get("type") == "_thread_finished":
                continue 
            
            await game_manager.broadcast(msg)
        except Exception as e:
            logging.error(f"Broadcast loop error: {e}")

@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await game_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "start_game":
                # 支持前端或消息里传 model 路径，否则用环境变量或默认路径
                model_path = data.get("model") or os.environ.get('MORTAL_MODEL') or '/data/mortal/mortal.pth'
                game_manager.start_game_thread(model_path)
            else:
                game_manager.action_queue.put(data)
    except WebSocketDisconnect:
        await game_manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        await game_manager.disconnect(websocket)
