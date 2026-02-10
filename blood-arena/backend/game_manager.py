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
        self.engine_type = "mjai-log" 
        self.action_queue = action_queue
        self.state_queue = state_queue
        self.shared_state = shared_state
        self.player_id = 0 
        self.ai_engine = ai_engine
        
        # Shadow State for Protocol Translation
        self.tehai = [] # List of strings: ["1m", "5z"]
        self.last_kawa = None # Tuple: (actor_id, tile_str)
        self.last_tsumo_tile = None # Latest tile drawn by self
        self.peng = [] # List of pon-ed tiles (e.g. ["1m", "5z"])

    def set_player_ids(self, ids):
        self.player_id = ids[0]
        logging.info(f"Human player ID set to {self.player_id}")

    def start_game(self, index):
        logging.info(f"Game {index} started")

    def update_state(self, game_index, events_json):
        """
        Called by libblood's set_scene for real-time updates.
        Also updates local shadow state.
        """
        try:
            events = json.loads(events_json)
            
            # Update Shadow State
            for ev in events:
                etype = ev.get("type")
                actor = ev.get("actor")
                
                if etype == "start_kyoku":
                    self.tehai = ev["tehais"][self.player_id]
                    self.last_kawa = None
                    self.last_tsumo_tile = None
                    self.peng = []
                    logging.info(f"Kyoku Start. Hand: {self.tehai}")
                    
                elif etype == "tsumo":
                    if actor == self.player_id:
                        pai = ev["pai"]
                        self.tehai.append(pai)
                        self.last_tsumo_tile = pai
                        
                elif etype == "dahai":
                    pai = ev["pai"]
                    self.last_kawa = (actor, pai)
                    if actor == self.player_id:
                        # Remove from hand (handle tsumogiri optimization if needed)
                        # We just remove the first matching instance to be safe
                        if pai in self.tehai:
                            self.tehai.remove(pai)
                        self.last_tsumo_tile = None # Discarded

                elif etype == "pon":
                    if actor == self.player_id:
                        consumed = ev["consumed"] # ["1m", "1m"]
                        for t in consumed:
                            if t in self.tehai:
                                self.tehai.remove(t)
                        # Track Peng for Kakan
                        self.peng.append(ev["pai"]) # Record the pon-ed tile
                    self.last_kawa = None # Consumed

                elif etype == "daiminkan": # Open Kan
                    if actor == self.player_id: # Call it
                        # Consumed 3 tiles from hand? No, Daiminkan consumes 3.
                        # Wait, Daiminkan is Calling a Kan. 
                        # It consumes 3 tiles from hand.
                        consumed = ev.get("consumed", []) 
                        for t in consumed: 
                             if t in self.tehai:
                                self.tehai.remove(t)
                    self.last_kawa = None
                
                elif etype == "kakan": # Added Kan
                     if actor == self.player_id:
                        pai = ev["pai"]
                        if pai in self.tehai:
                            self.tehai.remove(pai)
                        # Remove from peng
                        if pai in self.peng:
                            self.peng.remove(pai)
                            
                elif etype == "ankan": # Closed Kan
                     if actor == self.player_id:
                        consumed = ev["consumed"] # 4 tiles
                        for t in consumed:
                            if t in self.tehai:
                                self.tehai.remove(t)

            msg = {
                "type": "state_update",
                "data": {
                    "events": events,
                    "analysis": {}
                }
            }
            self.shared_state['latest'] = msg
            self.state_queue.put(msg)
        except Exception as e:
            logging.error(f"Error in update_state: {e}")

    # ... (rest of methods) ...

    def _translate_to_mjai(self, client_action, game_state):
        # ... (previous code) ...
        # (Inside Kan block)
            if act_type == "kan":
                # Daiminkan or Ankan/Kakan?
                if self.last_kawa and self.last_kawa[0] != actor_id:
                     # Daiminkan (Open Kan from discard)
                     target, pai = self.last_kawa
                     return {
                        "type": "daiminkan",
                        "actor": actor_id,
                        "target": target,
                        "pai": pai,
                        "consumed": [pai, pai, pai] 
                    }
                else:
                    # Ankan or Kakan (Self Kan)
                    from collections import Counter
                    counts = Counter(self.tehai)
                    
                    # 1. Check Kakan (Added Kan) - Priority? 
                    # If we have a Pon of X, and we have X in hand.
                    for p in self.peng:
                        if counts[p] >= 1:
                            return {
                                "type": "kakan",
                                "actor": actor_id,
                                "pai": p,
                                "consumed": [p, p, p] # consumed the pon?
                            }
                    
                    # 2. Check Ankan (4 in hand)
                    for t, c in counts.items():
                        if c == 4:
                             return {
                                "type": "ankan",
                                "actor": actor_id,
                                "consumed": [t, t, t, t]
                            }
                            
                    logging.warning("Kan requested but no candidate found.")
                    return {"type": "none"}


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

    # ... (react_batch remains mostly same, just calling new _translate_to_mjai) ...
    # Wait, need to preserve react_batch structure but allow it to use the new translate.

    def react_batch(self, game_states):
        """
        Called by libblood from Rust thread.
        Blocks until human action is received.
        """
        game_state = game_states[0] 
        events_json = game_state.events_json
        events = json.loads(events_json)
        
        # 2. Get AI Analysis
        analysis = {}
        mask = None
        if self.ai_engine:
            try:
                obs, mask = game_state.state.encode_obs(4, False)
                analysis = self._get_ai_analysis(game_state, obs, mask)
            except Exception as e:
                logging.error(f"AI Analysis failed: {e}")

        # 3. Determine Legal Actions
        is_interactive = False
        
        action_msgs = []
        is_interactive = False
        
        if mask is not None:
            # Check Ding Que
            if mask[31] or mask[32] or mask[33]:
                is_interactive = True
                action_msgs.append({"type": "ding_que"})
            
            # Check Actions (Pon/Kan/Hu)
            if mask[27] or mask[28] or mask[29]: 
                is_interactive = True
                actions_list = []
                if mask[27]: actions_list.append({"type": "pon"})
                if mask[28]: actions_list.append({"type": "kan"}) 
                if mask[29]: actions_list.append({"type": "hu"})
                
                action_msgs.append({ "type": "allow_actions", "actions": actions_list })
                
            # Check Discard
            if any(mask[0:27]):
                is_interactive = True
                
        # 4. State Update (FIRST, to set the scene)
        msg_state = {
            "type": "state_update",
            "data": { "events": events, "analysis": analysis }
        }
        self.shared_state['latest'] = msg_state
        self.state_queue.put(msg_state)

        # 4.5 Send Action Requests (SECOND, to override phase)
        for m in action_msgs:
            self.shared_state['latest'] = m
            self.state_queue.put(m)
        
        # 5. Handle Control Flow
        if is_interactive:
            logging.info("Waiting for human action...")
            action_data = self.action_queue.get()
            logging.info(f"Received human action: {action_data}")
            mjai_action = self._translate_to_mjai(action_data, game_state)
            logging.info(f"[DEBUG] react_batch submitting: {mjai_action}")
            return [json.dumps(mjai_action)]
        else:
            return [json.dumps({"type": "none"})]

    def _get_ai_analysis(self, game_state, obs, mask) -> Dict[str, Any]:
        """ Generate AI analysis. """
        import torch
        
        # Query AI
        with torch.no_grad():
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
            "best_action": {"type": "dahai", "pai": best_tile, "idx": best_idx}
        }

    def _translate_to_mjai(self, client_action, game_state):
        """ Convert simplified client action to full MJAI event using Shadow State. """
        atype = client_action.get("type")
        actor_id = self.player_id
        
        if atype == "ding_que":
            suit = client_action.get("suit")
            suit_map = {"m": "man", "p": "pin", "s": "sou"}
            suit_full = suit_map.get(suit, "man")
            ret = {"type": "ding_que", "actor": actor_id, "suit": suit_full}
            logging.info(f"[DEBUG] Sending DingQue to libblood: {ret}")
            return ret 
            
        if atype == "dahai":
            pai = client_action.get("pai")
            tsumogiri = (pai == self.last_tsumo_tile)
            return {
                "type": "dahai", 
                "actor": actor_id, 
                "pai": pai, 
                "tsumogiri": tsumogiri
            }

        if atype == "action":
            act_type = client_action["action"]["type"]
            
            if act_type == "pass":
                return {"type": "none"}
                
            if act_type == "hu":
                # Tsumo or Ron?
                if self.last_tsumo_tile: # If we just drew a tile, it's Tsumo
                    return {
                        "type": "hora", 
                        "actor": actor_id, 
                        "target": actor_id, 
                        "pai": self.last_tsumo_tile
                    }
                else: # Ron
                    target, pai = self.last_kawa if self.last_kawa else (0, "?")
                    return {
                        "type": "hora", 
                        "actor": actor_id, 
                        "target": target, 
                        "pai": pai
                    }

            if act_type == "pon":
                if not self.last_kawa:
                     logging.error("Pon requested but no last_kawa!")
                     return {"type": "none"}
                target, pai = self.last_kawa
                return {
                    "type": "pon",
                    "actor": actor_id,
                    "target": target,
                    "pai": pai,
                    "consumed": [pai, pai] # Consumes 2 matching tiles
                }
                
            if act_type == "kan":
                # Daiminkan or Ankan/Kakan?
                if self.last_kawa and self.last_kawa[0] != actor_id:
                     # Daiminkan (Open Kan from discard)
                     target, pai = self.last_kawa
                     return {
                        "type": "daiminkan",
                        "actor": actor_id,
                        "target": target,
                        "pai": pai,
                        "consumed": [pai, pai, pai] 
                    }
                else:
                    # Ankan or Kakan (Self Kan)
                    # We need to know WHICH tile to kan.
                    # Frontend sending just "kan" is ambiguous if multiple options.
                    # AI Analysis result usually has specific "kan" action index?
                    # Or we just find the first valid quad/triplet in hand.
                    
                    # Heuristic: Check for 4 same tiles (Ankan) or Triplet+Pon (Kakan)
                    # For MVP: Look for 4 copies in tehai.
                    from collections import Counter
                    counts = Counter(self.tehai)
                    
                    # Check Ankan (4 in hand)
                    for t, c in counts.items():
                        if c == 4:
                             return {
                                "type": "ankan",
                                "actor": actor_id,
                                "consumed": [t, t, t, t]
                            }
                            
                    # Check Kakan (1 in hand + 3 in Pon)
                    # We don't track 'peng' in shadow state yet, but we should.
                    # Fallback/TODO: If strict checking needed, add 'peng' list.
                    # For now, if we found nothing, maybe return None?
                    logging.warning("Kan requested but no obvious candidate found in hand.")
                    return {"type": "none"}

        logging.warning(f"Unhandled client action: {client_action}")
        return {"type": "none"} # Safety fallback



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
