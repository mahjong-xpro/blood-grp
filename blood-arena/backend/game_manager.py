import json
import asyncio
import threading
import queue
import logging
import random
from typing import List, Dict, Any, Optional
from fastapi import WebSocket

# Try to import libblood, handle failure for development environment without compiled module
try:
    from libblood import arena
except ImportError:
    logging.warning("libblood not found. AI opponents will not work.")
    arena = None

# 每局初始分（与 rules.md / libblood 一致），用于计算 8 局累计输赢
INITIAL_SCORE = 60000
MATCH_GAMES = 8  # 一场对局共 8 局


class _ChampionWithHumanObserver:
    """包装 AI 引擎：libblood 在 AI 回合调用 set_scene 时会尝试 update_state；
    人已和牌后不再轮到人，只有 AI 收到 set_scene，必须由此转发 state 给前端，否则界面卡死。"""

    def __init__(self, ai_engine, human_engine: "HumanEngine"):
        self._ai = ai_engine
        self._human = human_engine

    def update_state(self, game_index, events_json):
        self._human.update_state(game_index, events_json)

    def __getattr__(self, name):
        return getattr(self._ai, name)


class HumanEngine:
    def __init__(self, action_queue: queue.Queue, state_queue: queue.Queue, shared_state: Dict[str, Any]):
        self.name = "Human"
        self.engine_type = "mjai-log" 
        self.action_queue = action_queue
        self.state_queue = state_queue
        self.shared_state = shared_state
        self.player_id = 0
        
        # Shadow State for Protocol Translation
        self.tehai = [] # List of strings: ["1m", "5z"]
        self.last_kawa = None # Tuple: (actor_id, tile_str)
        self.last_tsumo_tile = None # Latest tile drawn by self
        self.peng = [] # List of pon-ed tiles (e.g. ["1m", "5z"])

        # 8 局制：累计输赢（相对每局初始分）、当前已完成局数（1..8）
        self.match_deltas = [0, 0, 0, 0]
        self.match_game_index = 0

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
            logging.info(f"[DEBUG] update_state events count: {len(events)}")
            # Optional: log specific event types to see if DingQue is there
            # dqs = [e for e in events if e['type'] == 'ding_que']
            # if dqs: logging.info(f"[DEBUG] DingQue events found: {dqs}")
            self.shared_state['latest'] = msg
            self.state_queue.put(msg)
        except Exception as e:
            logging.error(f"Error in update_state: {e}")

    def end_kyoku(self, index):
        logging.info(f"Kyoku {index} ended")

    def end_game(self, index, scores, final_tehais=None):
        """一局结束：累计本局得失分，推送 game_over；若已打满 8 局则标记 match_over。"""
        logging.info(f"Game {index} ended with scores: {scores}")
        # 本局相对初始分的得失
        deltas = [int(s) - INITIAL_SCORE for s in scores]
        for i in range(4):
            self.match_deltas[i] += deltas[i]
        self.match_game_index += 1
        is_match_over = self.match_game_index >= MATCH_GAMES

        msg = {
            "type": "game_over",
            "scores": list(scores),
            "match_deltas": list(self.match_deltas),
            "game_number": self.match_game_index,
            "is_match_over": is_match_over,
        }
        if final_tehais is not None and len(final_tehais) == 4:
            msg["tehais"] = final_tehais
        self.shared_state['latest'] = msg
        self.state_queue.put(msg)

    # ... (react_batch remains mostly same, just calling new _translate_to_mjai) ...
    # Wait, need to preserve react_batch structure but allow it to use the new translate.

    def _get_legal_actions(self, cans) -> List[Dict[str, Any]]:
        """ Convert Rust ActionCandidate to list of allowed client actions. """
        actions = []
        if cans.can_ding_que:
            actions.append({"type": "ding_que"})
        
        if cans.can_discard:
            actions.append({"type": "dahai"})
        
        # Interactive Actions
        if cans.can_pon: actions.append({"type": "pon"})
        # kan covers ankan, daiminkan, kakan
        if cans.can_kan: actions.append({"type": "kan"}) 
        # agari covers tsumo and ron
        if cans.can_agari: actions.append({"type": "hu"})
        
        if cans.can_pass: actions.append({"type": "pass"})
        
        return actions

    def react_batch(self, game_states):
        """
        Refactored: Strictly logic-driven interaction.
        """
        # 1. Get GameState Wrapper
        # game_states is List[GameState] from libblood
        # GameState has attributes: .state (PlayerState) and .events_json (str)
        wrapper = game_states[self.player_id] 
        events = json.loads(wrapper.events_json)
        
        # Get actual PlayerState object for logic query
        player_state = wrapper.state
        
        # 2. ME 玩家不需要 AI 分析提醒，不请求
        analysis = {}

        # 3. Determine Legal Actions (Logic Layer)
        try:
            cans = player_state.last_cans
            legal_actions = self._get_legal_actions(cans)
        except Exception as e:
            logging.error(f"Failed to get ActionCandidate: {e}")
            return [json.dumps({"type": "none"})]

        if not legal_actions:
            return [json.dumps({"type": "none"})]

        # 4. Send Action Request FIRST so user can click immediately (replay may have delays)
        msg_req = {
            "type": "action_request",
            "actions": legal_actions
        }
        self.shared_state['latest'] = msg_req
        self.state_queue.put(msg_req)

        # 5. Send Full State Update (Base Layer) - replay may have delays, user already has canDiscard
        msg_state = {
            "type": "state_update",
            "data": { "events": events, "analysis": analysis }
        }
        self.shared_state['latest'] = msg_state
        self.state_queue.put(msg_state)
        
        logging.info(f"Waiting for human action. Legal: {[a['type'] for a in legal_actions]}")

        # 6. Wait for Valid Action
        while True:
            action_data = self.action_queue.get()
            atype = action_data.get("type")
            
            # Validation: match atype against legal_actions
            is_valid = False
            for allowed in legal_actions:
                allowed_type = allowed["type"]
                if atype == allowed_type:
                    is_valid = True
                elif atype == "action": # Frontend action bar (pon/kan/hu/pass)
                    sub_act = action_data.get("action", {}).get("type")
                    if sub_act == allowed_type:
                        is_valid = True
            
            if is_valid:
                logging.info(f"Received valid action: {action_data}")
                mjai_action = self._translate_to_mjai(action_data, player_state)
                # Drain leftover actions so next turn does not consume a stale click
                try:
                    while True:
                        self.action_queue.get_nowait()
                except queue.Empty:
                    pass
                return [json.dumps(mjai_action)]
            
            # Invalid Fallback
            logging.warning(f"Ignored invalid action {atype}. Legal: {legal_actions}")
            # Resend State and Request to force sync
            self.state_queue.put(msg_state)
            self.state_queue.put(msg_req)

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
                # Use game_state.last_cans to decide type: only return the kan type that is actually allowed.
                cans = getattr(game_state, "last_cans", None)
                can_daiminkan = getattr(cans, "can_daiminkan", False) if cans else False
                can_kakan = getattr(cans, "can_kakan", False) if cans else False
                can_ankan = getattr(cans, "can_ankan", False) if cans else False

                if can_daiminkan and self.last_kawa and self.last_kawa[0] != actor_id:
                    target, pai = self.last_kawa
                    return {
                        "type": "daiminkan",
                        "actor": actor_id,
                        "target": target,
                        "pai": pai,
                        "consumed": [pai, pai, pai],
                    }
                if can_kakan:
                    from collections import Counter
                    counts = Counter(self.tehai)
                    try:
                        kakan_candidates = getattr(game_state, "kakan_candidates", None)
                        if callable(kakan_candidates):
                            kakan_candidates = kakan_candidates()
                        else:
                            kakan_candidates = []
                    except Exception:
                        kakan_candidates = []
                    for p in (kakan_candidates if kakan_candidates else self.peng):
                        if counts.get(p, 0) >= 1:
                            return {
                                "type": "kakan",
                                "actor": actor_id,
                                "pai": p,
                                "consumed": [p, p, p],
                            }
                    logging.warning("Kan: can_kakan true but no valid candidate found.")
                    return {"type": "none"}
                if can_ankan:
                    from collections import Counter
                    counts = Counter(self.tehai)
                    for t, c in counts.items():
                        if c == 4:
                            return {
                                "type": "ankan",
                                "actor": actor_id,
                                "consumed": [t, t, t, t],
                            }
                    logging.warning("Kan: can_ankan true but no quad in hand.")
                    return {"type": "none"}
                logging.warning("Kan requested but no can_daiminkan/can_kakan/can_ankan.")
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
            # Load model for AI（未找到文件或加载失败会用随机权重，AI 会非常弱）
            try:
                if os.path.exists(ai_model_path):
                    logging.info("Loading AI model from: %s", ai_model_path)
                    state = torch.load(ai_model_path, map_location='cpu', weights_only=True)
                    cfg = state['config']
                    version = cfg['control'].get('version', 4)
                    mortal = Brain(version=version, conv_channels=cfg['resnet']['conv_channels'], num_blocks=cfg['resnet']['num_blocks']).eval()
                    dqn = DQN(version=version).eval()
                    mortal.load_state_dict(state['mortal'])
                    dqn.load_state_dict(state['current_dqn'])
                    logging.info("AI model loaded successfully from %s", ai_model_path)
                else:
                    logging.warning(
                        "Model file not found at %s — using RANDOM weights (AI will be very weak). "
                        "Set MORTAL_MODEL or pass model path in start_game.",
                        ai_model_path,
                    )
                    version = 4
                    mortal = Brain(version=version, conv_channels=192, num_blocks=40).eval()
                    dqn = DQN(version=version).eval()
            except Exception as e:
                logging.warning("Failed to load model from %s: %s — using RANDOM weights.", ai_model_path, e)
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
            human = HumanEngine(self.action_queue, self.state_queue, self.shared_state)
            human.match_deltas = [0, 0, 0, 0]
            human.match_game_index = 0

            # AI 回合时也需向前端推送 state_update（人已和牌后不再轮到人，set_scene 只调 AI，否则前端卡死）
            champion = _ChampionWithHumanObserver(ai_engine, human)

            # 8 局制：连续打满 8 局，每局独立种子
            env = arena.OneVsThree(disable_progress_bar=True, log_dir=None)
            seed_base = random.getrandbits(32)
            for game_num in range(MATCH_GAMES):
                seed = (seed_base, game_num)
                logging.info("Match game %d/8 seed: %s", game_num + 1, seed)
                env.py_vs_py(human, champion, seed, 1)
            logging.info("Match finished. Final deltas: %s", human.match_deltas)
            
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
