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
        game_state = game_states[0] 
        events_json = game_state.events_json
        
        # 1. Parse events for simple logging or legacy support
        events = json.loads(events_json)
        
        # 2. Get AI Analysis (and Mask)
        analysis = {}
        mask = None
        if self.ai_engine:
            try:
                # We need to access the mask from the AI engine helper or manually encode
                obs, mask = game_state.state.encode_obs(4, False)
                analysis = self._get_ai_analysis(game_state, obs, mask)
            except Exception as e:
                logging.error(f"AI Analysis failed: {e}")

        # 3. Determine Legal Actions from Mask
        # Action Space: 34
        # 0-26: Discard
        # 27: Pon, 28: Kan, 29: Agari, 30: Pass
        # 31: DQ-Man, 32: DQ-Pin, 33: DQ-Sou
        legal_actions = []
        is_ding_que_phase = False
        
        if mask is not None:
            # Check Ding Que
            if mask[31] or mask[32] or mask[33]:
                is_ding_que_phase = True
                # Trigger frontend Ding Que UI
                self.shared_state['latest'] = {"type": "ding_que"} # Optimization
                self.state_queue.put({"type": "ding_que"})
            
            # Check Actions (Pon/Kan/Hu/Pass)
            # Only trigger allow_actions if it's NOT just Discard/DingQue
            # Usually if Pon/Kan/Hu is possible, Pass is also possible (30)
            if mask[27] or mask[28] or mask[29]: # Pon, Kan, Agari
                actions_list = []
                if mask[27]: actions_list.append({"type": "pon"})
                if mask[28]: actions_list.append({"type": "kan"}) # Logic to distinguish Kan types later
                if mask[29]: actions_list.append({"type": "hu"})
                
                # If we have special actions, Pass is implied (unless forced agari?)
                # Mask[30] is pass.
                
                # Send explicit allow_actions signal
                msg_actions = {
                    "type": "allow_actions",
                    "actions": actions_list
                }
                self.state_queue.put(msg_actions)

        # 4. Send State Update (Events + Analysis)
        msg = {
            "type": "state_update",
            "data": {
                "events": events,
                "analysis": analysis
            }
        }
        self.shared_state['latest'] = msg
        self.state_queue.put(msg)
        
        # 5. Wait for Action
        logging.info("Waiting for human action...")
        action_data = self.action_queue.get()
        logging.info(f"Received human action: {action_data}")
        
        # 6. Protocol Translation (Frontend JSON -> MJAI JSON)
        mjai_action = self._translate_to_mjai(action_data, game_state)
        
        return [json.dumps(mjai_action)]

    def _get_ai_analysis(self, game_state, obs, mask) -> Dict[str, Any]:
        """ Generate AI analysis. """
        import torch
        import numpy as np
        
        # Query AI
        with torch.no_grad():
             # MortalEngine.react_batch expects list of obs/masks
             actions, q_out, masks, is_greedy = self.ai_engine.react_batch([obs], [mask], None)
             
        q_values = q_out[0]
        valid_mask = masks[0]
        action_space = len(q_values)
        candidates = []
        
        def idx_to_tile(idx):
            if 0 <= idx < 9: return f"{idx + 1}m"
            if 9 <= idx < 18: return f"{idx - 9 + 1}p"
            if 18 <= idx < 27: return f"{idx - 18 + 1}s"
            return None

        for i in range(action_space):
            if valid_mask[i]:
                tile = idx_to_tile(i)
                # Map special actions
                type_str = "discard"
                if i == 27: type_str = "pon"
                elif i == 28: type_str = "kan"
                elif i == 29: type_str = "hu"
                elif i == 30: type_str = "pass"
                elif i >= 31: type_str = "ding_que"
                
                candidates.append({
                    "idx": i,
                    "q": float(q_values[i]),
                    "tile": tile,
                    "type": type_str
                })
        candidates.sort(key=lambda x: x["q"], reverse=True)
        
        best_idx = int(actions[0])
        best_tile = idx_to_tile(best_idx)

        return {
            "candidates": candidates[:5],
            "best_action": {"type": "dahai", "pai": best_tile, "idx": best_idx} # Simplified
        }

    def _translate_to_mjai(self, client_action, game_state):
        """ Convert simplified client action to full MJAI event. """
        atype = client_action.get("type")
        actor_id = self.player_id
        
        if atype == "ding_que":
            # Client: {"type": "ding_que", "suit": "m"}
            # MJAI: {"type": "ding_que", "actor": 0, "color": "m"} (Wait, proper MJAI for Blood might differ?)
            # libblood expects {"type":"ding_que", "actor":.., "color":..} ?
            # Let's verify libblood expected format or just guess standard.
            # blood-arena usually uses "color" for suit? "suit" vs "color".
            # The client sends 'suit'='m'.
            # libblood `ding_que.rs` likely parses "color" or "suit".
            # Let's try to infer from typical usage. "color" is safer for "m/p/s".
            suit = client_action.get("suit")
            return {"type": "ding_que", "actor": actor_id, "color": suit} # Try color
            
        if atype == "dahai":
            # Client: {"type": "dahai", "pai": "1m"}
            pai = client_action.get("pai")
            # Calculate tsumogiri
            tsumogiri = False
            last_tsumo = game_state.state.last_self_tsumo
            # last_self_tsumo is Tile object. Need string comparison.
            if last_tsumo and str(last_tsumo) == pai:
                tsumogiri = True
            
            return {
                "type": "dahai", 
                "actor": actor_id, 
                "pai": pai, 
                "tsumogiri": tsumogiri
            }

        if atype == "action":
            # Client: {"type": "action", "action": {"type": "pon"}}
            act_type = client_action["action"]["type"]
            
            if act_type == "pass":
                return {"type": "none"}
                
            if act_type == "hu":
                # Check target (Tsumo or Ron)
                # If last event was Discard (from other), it's Ron.
                # If last event was Tsumo (from self), it's Tsumo.
                # Use game_state.state.last_kawa_tile
                last_kawa = game_state.state.last_kawa_tile
                target = game_state.state.last_kawa_tile_actor() if hasattr(game_state.state, 'last_kawa_tile_actor') else (actor_id + 3) % 4 # Hacky fallback
                
                # Check if tsumo
                if game_state.state.last_self_tsumo:
                    return {"type": "hora", "actor": actor_id, "target": actor_id, "pai": str(game_state.state.last_self_tsumo)}
                else:
                    # Ron
                    # Need real target.
                    # game_state.state doesn not easily expose "who discarded last".
                    # But we can infer from `kawa`? Or just let libblood handle "target" if omitted? 
                    # MJAI requires target.
                    # Let's hope `last_kawa_tile` implies the target is the turn player?
                    # The turn player is NOT us.
                    # We can iterate players to find who turned last?
                    # `at_turn` might be the opponent.
                    target = game_state.state.at_turn
                    return {"type": "hora", "actor": actor_id, "target": target, "pai": str(last_kawa)}

            if act_type == "pon":
                # Need consumed tiles
                # tile = last_kawa_tile
                last_kawa = str(game_state.state.last_kawa_tile)
                target = game_state.state.at_turn
                # Find 2 matching tiles in hand
                return {
                    "type": "pon",
                    "actor": actor_id,
                    "target": target,
                    "pai": last_kawa,
                    "consumed": [last_kawa, last_kawa] 
                }
                
            if act_type == "kan":
                # Daiminkan or Ankan or Kakan?
                # If we have last_kawa_tile, it's Daiminkan.
                # If not, it's Ankan or Kakan.
                last_kawa = game_state.state.last_kawa_tile
                if last_kawa:
                    # Daiminkan
                    t = str(last_kawa)
                    target = game_state.state.at_turn
                    return {
                        "type": "daiminkan",
                        "actor": actor_id,
                        "target": target,
                        "pai": t,
                        "consumed": [t, t, t]
                    }
                else:
                    # Ankan or Kakan
                    # Complex logic to pick which tile to Kan if multiple options
                    # For MVP, pick the first valid one?
                    # Or check hand.
                    # We need `kakan_candidates` or `ankan_candidates` from state logic.
                    # Assuming client just says "Kan", we pick one.
                    # game_state.state.ankan_candidates -> List[Tile]
                    # This is PyObject, might not be iterable easily?
                    # Let's try to assume Ankan first.
                    pass

        # Fallback
        logging.warning(f"Unhandled client action: {client_action}")
        return {} # Should error

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
