/**
 * Blood Arena Frontend Logic
 * Supports: Phase Management (DingQue/Playing), Action Buttons, AI Suggestions
 */
import { createApp, reactive, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';

const TILE_BASE = '/static/images/tiles';
function tileSrc(tile) {
    if (!tile || tile === 'back') return `${TILE_BASE}/Back.png`;
    const n = tile[0], s = tile[1];
    const suit = { m: 'Man', p: 'Pin', s: 'Sou' }[s] || 'Man';
    return `${TILE_BASE}/${suit}${n}.png`;
}

const app = createApp({
    setup() {
        // --- State ---
        const state = reactive({
            connected: false,
            gaming: false,
            phase: 'idle', // idle, dingque, playing, result
            gameEnded: false,

            myPlayerId: 0,
            currentActor: -1,
            tilesLeft: 108,

            // Game Data
            scores: [25000, 25000, 25000, 25000],
            tehai: [], // My hand
            tsumoTile: null, // Last drawn tile
            discards: [[], [], [], []],
            agari: [false, false, false, false],
            dingque: [null, null, null, null], // m, p, s for each player

            // Interactive
            validActions: [], // [{type: 'pon', ...}]
        });

        const analysis = reactive({ best_action: null });
        const ui = reactive({ selectedIdx: -1 });
        let ws = null;

        // --- Computed ---
        const isMyTurn = computed(() => state.currentActor === state.myPlayerId);

        // --- Network ---
        const connect = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);

            ws.onopen = () => { state.connected = true; };
            ws.onclose = () => {
                state.connected = false;
                setTimeout(connect, 3000);
            };
            ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
        };

        const send = (data) => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
        };

        // --- Message Handling ---
        function handleMessage(msg) {
            // console.log('WS Msg:', msg); // Verbose
            if (msg.type === 'state_update') {
                updateFullState(msg.data);
            } else if (msg.type === 'ding_que') {
                // Server asking for Ding Que
                state.phase = 'dingque';
            } else if (msg.type === 'allow_actions') {
                // Server offering actions (Pon/Kan/Hu)
                state.validActions = msg.actions;
            } else if (msg.type === 'game_over') {
                state.gameEnded = true;
                state.phase = 'result';
                state.gaming = false;
                alert(`Game Over! Scores: ${msg.scores.join(', ')}`);
            }
        }

        function updateFullState(data) {
            // Mapping complex server state to frontend state
            if (data.analysis) {
                analysis.best_action = data.analysis.best_action;
            }
            if (data.events) {
                replayEvents(data.events);
            }
        }

        // Helper
        function sortTiles(tiles) {
            const suitOrder = { 'm': 0, 'p': 1, 's': 2, 'z': 3 };
            tiles.sort((a, b) => {
                const suitA = a[1], suitB = b[1];
                const numA = parseInt(a[0]), numB = parseInt(b[0]);
                if (suitOrder[suitA] !== suitOrder[suitB]) return suitOrder[suitA] - suitOrder[suitB];
                return numA - numB;
            });
        }

        // --- Event Replay (Reconstruct State) ---
        function replayEvents(events) {
            // IMPORTANT: Do NOT wipe state here.
            // If backend sends incremental updates (e.g. only new events), wiping would destroy the hand.
            // State reset should only happen on 'start_game' or 'start_kyoku'.

            for (const ev of events) {
                switch (ev.type) {
                    case 'start_game':
                        state.gaming = true;
                        state.gameEnded = false;
                        state.scores = [25000, 25000, 25000, 25000];
                        // Don't force phase reset here, wait for flow
                        break;
                    case 'start_kyoku':
                        // Reset Round State
                        // Fix for One Tile Hand: Use tehais array if available
                        state.tehai = (ev.tehais ? ev.tehais[state.myPlayerId] : []) || ev.tehai || ev.hand || [];
                        sortTiles(state.tehai);
                        state.tsumoTile = null;
                        state.discards = [[], [], [], []];
                        state.agari = [false, false, false, false];
                        state.dingque = [null, null, null, null];

                        // Correct Tile Count (108 - 13*4 = 56)
                        state.tilesLeft = 56;

                        // Phase Stability: only 'playing' if we don't know better.
                        // Ideally, we wait for 'ding_que' message.
                        if (state.phase !== 'dingque') {
                            state.phase = 'playing';
                        }
                        break;
                    case 'ding_que':
                        // If we see a ding_que EVENT in history, it means someone declared it
                        if (ev.actor !== undefined && (ev.suit || ev.color)) {
                            state.dingque[ev.actor] = ev.suit || ev.color;
                        }
                        break;
                    case 'tsumo':
                        state.currentActor = ev.actor;
                        state.phase = 'playing'; // Resume playing

                        state.tilesLeft = Math.max(0, state.tilesLeft - 1);
                        if (ev.actor === state.myPlayerId && ev.pai) {
                            state.tsumoTile = ev.pai;
                        }
                        break;
                    case 'dahai':
                        state.currentActor = (ev.actor + 1) % 4; // Speculative next
                        state.phase = 'playing'; // Resume playing

                        if (!state.discards[ev.actor]) state.discards[ev.actor] = [];
                        state.discards[ev.actor].push(ev.pai);

                        if (ev.actor === state.myPlayerId) {
                            if (state.tsumoTile === ev.pai) {
                                state.tsumoTile = null;
                            } else {
                                const idx = state.tehai.indexOf(ev.pai);
                                if (idx > -1) state.tehai.splice(idx, 1);
                                // If we tsumogiri'd from hand, move tsumo to hand
                                if (state.tsumoTile) {
                                    state.tehai.push(state.tsumoTile);
                                    sortTiles(state.tehai); // Proper sort
                                    state.tsumoTile = null;
                                }
                            }
                        }
                        ui.selectedIdx = -1;
                        state.validActions = []; // Clear actions on new move
                        break;
                    case 'pon':
                    case 'kan':
                    case 'ankan':
                    case 'daiminkan':
                    case 'kakan':
                        state.currentActor = ev.actor;
                        state.validActions = [];

                        if (ev.actor === state.myPlayerId) {
                            // 1. Determine tiles to remove
                            let toRemove = [];
                            if (ev.type === 'kakan') {
                                // Added Kan: removes the specific added tile (pai)
                                if (ev.pai) toRemove.push(ev.pai);
                            } else {
                                // Pon/Daiminkan/Ankan: remove 'consumed' tiles
                                if (ev.consumed) toRemove = [...ev.consumed];
                            }

                            // 2. Remove them from hand/tsumo
                            for (const t of toRemove) {
                                if (state.tsumoTile === t) {
                                    state.tsumoTile = null;
                                } else {
                                    const idx = state.tehai.indexOf(t);
                                    if (idx > -1) state.tehai.splice(idx, 1);
                                }
                            }

                            // 3. Consolidate Tsumo (if we Ankan'd using hand tiles, Tsumo might still be there)
                            // Actually, if we Pon/Daiminkan, we shouldn't have a Tsumo tile usually (turn change).
                            // But for Ankan/Kakan, we do.
                            if (state.tsumoTile) {
                                state.tehai.push(state.tsumoTile);
                                state.tsumoTile = null;
                            }

                            // 4. Sort
                            sortTiles(state.tehai);
                        }
                        break;
                    case 'agari':
                        state.agari[ev.actor] = true;
                        break;
                }
            }
        }

        // --- Interaction ---
        function startGame() {
            send({ type: 'start_game' });
        }

        function doDingQue(suit) {
            send({ type: 'ding_que', suit: suit });
            state.phase = 'playing'; // Assume done, optimistic update
        }

        function onTileClick(tile, idx) {
            // Only allow click if my turn and playing
            if (!isMyTurn.value || state.phase !== 'playing') return;

            // Toggle selection
            if (ui.selectedIdx === idx) {
                // Confirm discard
                send({ type: 'dahai', actor: state.myPlayerId, pai: tile });
                state.tsumoTile = null; // Client side optimistic update
                ui.selectedIdx = -1;
            } else {
                ui.selectedIdx = idx;
            }
        }

        function doAction(act) {
            send({ type: 'action', action: act });
            state.validActions = [];
        }

        function actionLabel(type) {
            const map = { 'hu': '胡', 'pon': '碰', 'kan': '杠', 'pass': '过' };
            return map[type] || type.toUpperCase();
        }

        function dingqueLabel(suit) {
            const map = {
                'm': '万', 'p': '筒', 's': '条',
                'man': '万', 'pin': '筒', 'sou': '条'
            };
            return map[suit] || suit;
        }

        // --- Helpers ---
        function player(offset) {
            const id = (state.myPlayerId + offset) % 4;
            return {
                score: state.scores[id],
                dingque: state.dingque[id],
                agari: state.agari[id]
            };
        }

        function hand(offset) {
            // For now only show my hand
            if (offset === 0) return state.tehai;
            return [];
        }

        function discards(offset) {
            const id = (state.myPlayerId + offset) % 4;
            return state.discards[id] || [];
        }

        function isRecommended(tile) {
            return analysis.best_action && analysis.best_action.pai === tile;
        }

        // Init
        connect();

        return {
            state, analysis, ui, isMyTurn,
            startGame, doDingQue, onTileClick, doAction,
            tileSrc, player, hand, discards, isRecommended, actionLabel
        };
    }
});

app.mount('#app');
