import json
import asyncio
import threading
import queue
import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

# Try to import libblood, handle failure for development environment without compiled module
try:
    from libblood import arena
except ImportError:
    logging.warning("libblood not found. AI opponents will not work.")
    arena = None

class HumanEngine:
    def __init__(self, action_queue: queue.Queue, state_queue: queue.Queue, shared_state: Dict[str, Any], ai_engine=None):
        self.name = "Human"
        # Trick libblood to use MjaiLogBatchAgent, which passes full event logs
        self.engine_type = "mjai-log" 
        self.action_queue = action_queue
        self.state_queue = state_queue
        self.shared_state = shared_state
        self.player_id = 0 # Will be set by set_player_ids
        self.ai_engine = ai_engine

    def set_player_ids(self, ids):
        self.player_id = ids[0]
        logging.info(f"Human player ID set to {self.player_id}")

    def start_game(self, index):
        logging.info(f"Game {index} started")

    def end_kyoku(self, index):
        logging.info(f"Kyoku {index} ended")

    def end_game(self, index, scores):
        logging.info(f"Game {index} ended with scores: {scores}")
        # Send end game signal to UI
        msg = {
            "type": "game_over",
            "scores": scores
        }
        self.shared_state['latest'] = msg
        self.state_queue.put(msg)

    def react_batch(self, game_states):
        """
        Called by libblood from Rust thread.
        Blocks until human action is received.
        """
        # Assuming batch size 1 for 1v3
        game_state = game_states[0] 
        events_json = game_state.events_json
        
        # 1. Parse events and reconstruct UI state
        events = json.loads(events_json)
        # ui_state = self._reconstruct_state(events) # Deprecated state reconstruction
        
        # 2. Get AI Analysis if available
        analysis = {}
        if self.ai_engine:
            try:
                analysis = self._get_ai_analysis(game_state)
            except Exception as e:
                logging.error(f"AI Analysis failed: {e}")

        # 3. Send state to WebSocket handler (consumer of state_queue)
        # We pass raw events now, letting frontend handle state (or we could still use _reconstruct_state for legacy support)
        # For Phase 3, we want to rely on events, but let's keep it simple: just send events + analysis
        msg = {
            "type": "state_update",
            "data": {
                "events": events,
                "analysis": analysis
            }
        }
        self.shared_state['latest'] = msg # Cache for reconnection
        self.state_queue.put(msg)
        
        # 4. Block and wait for action from WebSocket handler (producer of action_queue)
        logging.info("Waiting for human action...")
        action_data = self.action_queue.get()
        logging.info(f"Received human action: {action_data}")
        
        # 5. Return action as JSON string (format expected by MjaiLogBatchAgent)
        return [json.dumps(action_data)]

    def _get_ai_analysis(self, game_state) -> Dict[str, Any]:
        """
        Generate AI analysis for the current state.
        """
        import torch
        import numpy as np
        
        # 1. Encode Observation
        # version=4 is standard for current Mortal
        obs, mask = game_state.state.encode_obs(4, False)
        
        # 2. Convert to Tensor (Batch size 1)
        obs_tensor = torch.as_tensor(np.stack([obs], axis=0))
        mask_tensor = torch.as_tensor(np.stack([mask], axis=0))
        invisible_obs = None # Oracle not used for human hints usually, or maybe we want to cheat? Let's stick to normal AI.
        
        # 3. Query AI
        # MortalEngine.react_batch returns: actions, q_out, masks, is_greedy
        with torch.no_grad():
             # We need to access the internal _react_batch or similar logic because react_batch expects list of obs
             # But MortalEngine.react_batch handles list inputs.
             # Note: MortalEngine.react_batch needs 'obs' as list or numpy array.
             # We passed tensor, let's pass numpy array to be safe as per engine.py code
             actions, q_out, masks, is_greedy = self.ai_engine.react_batch([obs], [mask], None)
             
        # 4. Process Q-values
        # q_out is [batch, action_space]
        q_values = q_out[0]
        valid_mask = masks[0]
        
        # Calculate Win Rate (Sigmoid of Q-value? Or just raw Q if it's expected reward?)
        # Mortal v4 Q-values are likely Expected Game Result (Rank/Score normalized).
        # We can just show the top actions.
        
        # Find best actions
        action_space = len(q_values)
        candidates = []
        
        def idx_to_tile(idx):
            if 0 <= idx < 9: return f"{idx + 1}m"
            if 9 <= idx < 18: return f"{idx - 9 + 1}p"
            if 18 <= idx < 27: return f"{idx - 18 + 1}s"
            return None

        for i in range(action_space):
            if valid_mask[i]:
                # Determine action type and tile
                tile = idx_to_tile(i)
                action_type = "discard" if i < 27 else "other"
                
                candidates.append({
                    "idx": i,
                    "q": float(q_values[i]), # Ensure float for JSON serialization
                    "tile": tile,
                    "type": action_type
                })
        candidates.sort(key=lambda x: x["q"], reverse=True)
        
        best_idx = int(actions[0])
        best_tile = idx_to_tile(best_idx)

        # Construct best_action object similar to MJAI event for easy matching
        best_action_obj = {
            "type": "dahai" if best_idx < 27 else "other",
            "pai": best_tile,
            "idx": best_idx
        }
        
        return {
            "candidates": candidates[:5], # Top 5
            "best_action": best_action_obj
        }

    def _reconstruct_state(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Legacy: not used in Phase 3 message format, but keeping for reference if needed
        return {"events": events}

class GameManager:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.state_queue = queue.Queue()
        self.active_connections: List[WebSocket] = []
        self.shared_state = {} # Stores 'latest' message
        self.thread = None
        self.running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logging.info(f"Client connected. Total: {len(self.active_connections)}")
        if 'latest' in self.shared_state:
            await websocket.send_json(self.shared_state['latest'])

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logging.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        self.shared_state['latest'] = message
        # Broadcast to all active connections
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logging.error(f"Error broadcasting to client: {e}")

    def start_game_thread(self, ai_model_path: str):
        if self.running:
            return
        # Clear stale actions
        try:
            while True:
                self.action_queue.get_nowait()
        except queue.Empty:
            pass
        self.running = True
        self.thread = threading.Thread(target=self._run_libblood, args=(ai_model_path,))
        self.thread.start()
        logging.info("Game thread started")

    def _run_libblood(self, ai_model_path: str):
        # ... (setup code remains same) ...
        # But we need an async loop to call broadcast?
        # No, _run_libblood is in a THREAD.
        # We cannot call async methods describing websocket directly.
        # We need a thread-safe way to trigger broadcast in the main loop.
        # OPTION: Use `asyncio.run_coroutine_threadsafe` if we have reference to loop.
        # OR: Keep `state_queue` but make `main.py` handle broadcasting.
        pass

# ...
# Actually, let's keep it simple.
# Restore `state_queue` but make `main.py` consume it ONCE and broadcast to ALL.
# The previous design had the CONSUMER inside the connection handler.
# That means each connection consumed one item. THAT WAS THE BUG.
# Only ONE connection (the first one) got the message.
# WE NEED A SINGLE BACKGROUND TASK IN MAIN.PY TO CONSUME QUEUE AND BROADCAST.

    def start_game_thread(self, ai_model_path: str):
        if self.running:
            return
        # Clear stale actions from any previous session so new game doesn't consume them
        try:
            while True:
                self.action_queue.get_nowait()
        except queue.Empty:
            pass
        self.running = True
        self.thread = threading.Thread(target=self._run_libblood, args=(ai_model_path,))
        self.thread.start()
        logging.info("Game thread started")

    def _run_libblood(self, ai_model_path: str):
        try:
            if not arena:
                logging.error("Arena not available")
                return

            # Initialize AI Engine (Mortal) FIRST
            import torch
            from mortal.model import Brain, DQN
            from mortal.engine import MortalEngine
            import os

            ai_engine = None
            # Load model for AI
            try:
                if os.path.exists(ai_model_path):
                    state = torch.load(ai_model_path, map_location='cpu', weights_only=True)
                    cfg = state['config']
                    version = cfg['control'].get('version', 4)
                    
                    mortal = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'], num_blocks=cfg['resnet']['num_blocks']).eval()
                    dqn = DQN(version=version).eval()
                    mortal.load_state_dict(state['mortal'])
                    dqn.load_state_dict(state['current_dqn'])
                else:
                     logging.warning(f"Model file not found: {ai_model_path}, using random init")
                     version = 4
                     mortal = Brain(version=version, conv_channels=192, num_blocks=40).eval()
                     dqn = DQN(version=version).eval()
            except Exception as e:
                logging.warning(f"Failed to load model: {e}")
                # Random init fallback
                version = 4
                mortal = Brain(version=version, conv_channels=192, num_blocks=40).eval()
                dqn = DQN(version=version).eval()
                
            ai_engine = MortalEngine(
                mortal, dqn, is_oracle=False, version=version, 
                device=torch.device('cpu'), # Use CPU for inference to fit on standard machines
                enable_rule_based_agari_guard=True,
                name="MortalAI"
            )

            # Initialize Human Engine with AI Engine injected
            human = HumanEngine(self.action_queue, self.state_queue, self.shared_state, ai_engine=ai_engine)
            
            # Setup 1v3 Arena
            # Seed doesn't matter much for human play
            env = arena.OneVsThree(disable_progress_bar=True, log_dir=None)
            
            # Run 1 game
            # human is Challenger (Player 0 usually)
            # ai_engine is Champion
            env.py_vs_py(
                human,      # Challenger (Python Object)
                ai_engine,  # Champion (Python Object)
                (12345, 0), # Seed
                1,          # 1 game
            )
            logging.info("Game finished.")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            logging.error(f"Error in game execution: {e}", exc_info=True)
        finally:
            self.running = False
            # Let WebSocket sender exit instead of blocking forever on state_queue.get()
            try:
                self.state_queue.put({"type": "_thread_finished"}, block=False)
            except Exception:
                pass
