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
            console.log('WS Msg:', msg);
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
            // In a real app, this might be partial updates.
            // Here we mostly rely on event replay or direct data.

            if (data.analysis) {
                analysis.best_action = data.analysis.best_action;
            }
            if (data.events) {
                replayEvents(data.events);
            }
        }

        // --- Event Replay (Reconstruct State) ---
        function replayEvents(events) {
            // Reset per-kyoku state if start_kyoku
            // This logic matches previous arena.js but expanded

            for (const ev of events) {
                switch (ev.type) {
                    case 'start_game':
                        state.gaming = true;
                        state.gameEnded = false;
                        state.phase = 'playing'; // Default, might switch to dingque later
                        state.scores = [25000, 25000, 25000, 25000];
                        break;
                    case 'start_kyoku':
                        state.tehai = ev.tehai || [];
                        state.tsumoTile = null;
                        state.discards = [[], [], [], []];
                        state.agari = [false, false, false, false];
                        state.dingque = [null, null, null, null];
                        state.tilesLeft = 108; // Approx (Blood is 108)
                        state.phase = 'playing';
                        break;
                    case 'ding_que':
                        // ev.choices might inform us, but usually we wait for active request
                        // If event is "ding_que_done", update badges
                        break;
                    case 'tsumo':
                        state.currentActor = ev.actor;
                        state.tilesLeft--;
                        if (ev.actor === state.myPlayerId && ev.pai) {
                            state.tsumoTile = ev.pai;
                        }
                        break;
                    case 'dahai':
                        state.currentActor = (ev.actor + 1) % 4; // Speculative next
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
                                    state.tehai.sort(); // Simple sort
                                    state.tsumoTile = null;
                                }
                            }
                        }
                        ui.selectedIdx = -1;
                        state.validActions = []; // Clear actions on new move
                        break;
                    case 'pon':
                    case 'kan':
                        state.currentActor = ev.actor;
                        state.validActions = [];
                        // Remove from hand implementation (simplified)
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
            state.phase = 'playing'; // Assume done, wait for server
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
