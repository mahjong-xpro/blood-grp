import json
import asyncio
import threading
import queue
import logging
from typing import List, Dict, Any, Optional

# Try to import libblood, handle failure for development environment without compiled module
try:
    from libblood import arena
except ImportError:
    logging.warning("libblood not found. AI opponents will not work.")
    arena = None

class HumanEngine:
    def __init__(self, action_queue: queue.Queue, state_queue: queue.Queue, shared_state: Dict[str, Any]):
        self.name = "Human"
        # Trick libblood to use MjaiLogBatchAgent, which passes full event logs
        self.engine_type = "mjai-log" 
        self.action_queue = action_queue
        self.state_queue = state_queue
        self.shared_state = shared_state
        self.player_id = 0 # Will be set by set_player_ids

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
        ui_state = self._reconstruct_state(events)
        
        # 2. Send state to WebSocket handler (consumer of state_queue)
        msg = {
            "type": "state_update",
            "data": ui_state
        }
        self.shared_state['latest'] = msg # Cache for reconnection
        self.state_queue.put(msg)
        
        # 3. Block and wait for action from WebSocket handler (producer of action_queue)
        logging.info("Waiting for human action...")
        action_data = self.action_queue.get()
        logging.info(f"Received human action: {action_data}")
        
        # 4. Return action as JSON string (format expected by MjaiLogBatchAgent)
        return [json.dumps(action_data)]

    def _reconstruct_state(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse MJAI events to build a state object for the Frontend UI.
        This is a simplified re-implementation of state tracking.
        """
        # Basic state structure
        tehai = []
        discards = [[], [], [], []]
        melds = [[], [], [], []]
        scores = [60000] * 4  # 血战到底规则初始分
        current_turn = 0
        
        # Replay all events to build current state
        # In a real impl, we might want to cache this or use libblood's state if exposed
        for ev in events:
            ev_type = ev.get("type")
            actor = ev.get("actor")
            
            if ev_type == "start_kyoku":
                tehai = ev.get("tehai", [])
                scores = ev.get("scores", scores)
            
            elif ev_type == "tsumo":
                if actor == self.player_id:
                    tile = ev.get("pai")
                    tehai.append(tile)
            
            elif ev_type == "dahai":
                pai = ev.get("pai")
                discards[actor].append(pai)
                if actor == self.player_id and pai in tehai:
                    tehai.remove(pai) # Simple removal, distinct tiles might need handling
                    
            elif ev_type == "pon" or ev_type == "chi" or ev_type == "daiminkan":
                # Handle melds (simplified)
                melds[actor].append(ev)
                # Remove consumed tiles from hand if it's us
                if actor == self.player_id:
                     consumed = ev.get("consumed", [])
                     for t in consumed:
                         # For pon/daiminkan, the target tile is not in hand, only 2/3 are
                         # But 'consumed' usually includes the target tile? 
                         # Creating a robust state tracker is complex.
                         # For MVP, we might rely on 'events' log and let frontend replay it?
                         pass
            
            # ... Handle other events
            
        # For Phase 1 MVP: Just send the raw events to frontend
        # The frontend logic (reused from log-viewer) is good at replaying logs!
        return {
            "events": events,
            # We can compute valid actions here if needed, or rely on frontend
            # Actually, backend needs to tell frontend what actions are VALID right now.
            # But MjaiLogBatchAgent doesn't give us list of valid actions easily.
            # We might need to rely on the fact that MjaiLogBatchAgent expects *any* valid MJAI event.
        }

class GameManager:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.state_queue = queue.Queue()
        self.shared_state = {} # Stores 'latest' message
        self.thread = None
        self.running = False

    def start_game_thread(self, ai_model_path: str):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_libblood, args=(ai_model_path,))
        self.thread.start()
        logging.info("Game thread started")

    def _run_libblood(self, ai_model_path: str):
        try:
            if not arena:
                logging.error("Arena not available")
                return

            # Initialize Human Engine
            human = HumanEngine(self.action_queue, self.state_queue, self.shared_state)
            
            # Initialize AI Engine (Mortal)
            import torch
            from mortal.model import Brain, DQN
            from mortal.engine import MortalEngine

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
                     raise FileNotFoundError(f"Model file not found: {ai_model_path}")

            except Exception as e:
                logging.warning(f"Failed to load model: {e}")
                logging.info("Initializing random model for testing.")
                # Random init
                version = 4
                mortal = Brain(version=version, conv_channels=192, num_blocks=40).eval()
                dqn = DQN(version=version).eval()
                
            ai_engine = MortalEngine(
                mortal, dqn, is_oracle=False, version=version, 
                device=torch.device('cpu'), # Use CPU for inference to fit on standard machines
                enable_rule_based_agari_guard=True,
                name="MortalAI"
            )
            
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
