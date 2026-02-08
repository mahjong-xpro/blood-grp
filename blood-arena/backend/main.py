import sys
import os
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import queue
import asyncio

# Setup paths to import sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Mahjong/blood
sys.path.append(parent_dir)

# Set MORTAL_CFG for mortal module
os.environ['MORTAL_CFG'] = os.path.join(parent_dir, 'mortal', 'config.toml')

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Import GameManager
from backend.game_manager import GameManager

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
app.mount("/css", StaticFiles(directory="frontend/css"), name="css")

# Initialize Game Manager
game_manager = GameManager()

@app.get("/")
async def get():
    return FileResponse("frontend/index_v2.html")

# Keeping legacy route for reference if needed
@app.get("/legacy")
async def get_legacy():
    return FileResponse("frontend/index.html")

@app.websocket("/ws/game")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Send latest state if available (Reconnection support)
    if 'latest' in game_manager.shared_state:
        await websocket.send_json(game_manager.shared_state['latest'])
    
    # Register this websocket with the game manager (or just use it directly here)
    # GameManager is designed to run in a separate thread and communicate via queue
    
    # Start the game thread if not running
    # TODO: Pass actual model path
    # For MVP, we assume a default model path or let it fail gently
    model_path = os.path.join(parent_dir, "mortal", "models", "best.pth") 
    
    game_manager.start_game_thread(model_path)
    
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
                    # Already started thread
                    continue
                
                # Verify if it's a valid acton
                # For MVP, just put in queue
                game_manager.action_queue.put(data)
        except WebSocketDisconnect:
            logging.info("Client disconnected")
        except Exception as e:
            logging.error(f"Receiver error: {e}")

    # Run both
    await asyncio.gather(sender(), receiver())
