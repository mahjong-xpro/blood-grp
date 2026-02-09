import asyncio
import websockets
import json
import requests
import time

async def verify_game():
    uri = "ws://localhost:8000/ws/game"
    
    # 1. Start Game
    print("Starting game...")
    try:
        # Add timeout to prevent hanging
        resp = requests.post("http://localhost:8000/start_game?ai_model=/Users/twosson/Mahjong/blood/data/models/latest.pth", timeout=5)
        print(f"Start Game Response: {resp.status_code}")
    except requests.exceptions.Timeout:
        print("Start game request timed out (Game might have started anyway)")
    except Exception as e:
        print(f"Failed to start game: {e}")
        # Continue to connect anyway, as game might be running

    # 2. Connect to WebSocket
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")
            
            #Wait for state update
            start_time = time.time()
            while time.time() - start_time < 15: # 15s total timeout
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "state_update":
                        print("Received state_update")
                        # Verify AI Analysis
                        analysis = data.get("data", {}).get("analysis", {})
                        if analysis:
                            print("✅ AI Analysis Found!")
                            print(f"Best Action: {analysis.get('best_action')}")
                            print(f"Candidates Count: {len(analysis.get('candidates', []))}")
                            return
                        else:
                            print("⚠️ State update received but NO AI analysis found yet.")
                    
                except asyncio.TimeoutError:
                    print("Waiting for message...")
                    continue
                except Exception as e:
                    print(f"Error receiving: {e}")
                    break
            print("❌ Verification Timed Out")
    except Exception as e:
        print(f"WebSocket Connection Failed: {e}")

if __name__ == "__main__":
    asyncio.run(verify_game())
