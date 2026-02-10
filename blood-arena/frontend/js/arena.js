/**
 * Blood Arena Frontend Logic
 * Supports: Phase Management, Action Buttons, AI Suggestions, Melds (Fuuro)
 */
import { createApp, reactive, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';

const TILE_BASE = '/static/images/tiles';
const AUDIO_BASE = '/static/audio';
function tileSrc(tile) {
    if (!tile || tile === 'back' || tile === '?') return `${TILE_BASE}/Back.png`;
    const n = tile[0], s = tile[1];
    const suit = { m: 'Man', p: 'Pin', s: 'Sou' }[s] || 'Man';
    return `${TILE_BASE}/${suit}${n}.png`;
}
function playTileSound(pai) {
    if (!pai || pai === '?') return;
    try {
        const audio = new Audio(`${AUDIO_BASE}/${pai}.m4a`);
        audio.volume = 0.6;
        audio.play().catch(() => {});
    } catch (_) {}
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
            dingque: [null, null, null, null], // m, p, s
            fuuro: [[], [], [], []], // Melds

            // Interactive
            validActions: [],
            canDiscard: false,
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
            if (msg.type === 'state_update') {
                updateFullState(msg.data);
            } else if (msg.type === 'action_request') {
                handleActionRequest(msg.actions);
            } else if (msg.type === 'game_over') {
                state.gameEnded = true;
                state.phase = 'result';
                state.gaming = false;
                if (msg.scores && msg.scores.length === 4) {
                    state.scores = [...msg.scores];
                }
            }
        }

        function handleActionRequest(actions) {
            console.log("Action Request:", actions);
            state.gaming = true; // we are in a game when backend asks for an action
            state.validActions = [];
            state.canDiscard = false;

            let isDingQue = false;
            let isDiscard = false;

            for (const act of actions) {
                if (act.type === 'ding_que') isDingQue = true;
                if (act.type === 'dahai') isDiscard = true;
                if (['pon', 'kan', 'hu', 'pass'].includes(act.type)) {
                    state.validActions.push(act);
                }
            }

            if (isDingQue) {
                state.phase = 'dingque';
            } else {
                state.phase = 'playing';
                if (isDiscard) {
                    state.canDiscard = true;
                    state.currentActor = state.myPlayerId;
                }
            }
        }

        function updateFullState(data) {
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
                if (a === '?' || b === '?') return 0; // Don't sort unknowns
                const suitA = a[1], suitB = b[1];
                const numA = parseInt(a[0]), numB = parseInt(b[0]);
                if (suitOrder[suitA] !== suitOrder[suitB]) return suitOrder[suitA] - suitOrder[suitB];
                return numA - numB;
            });
        }

        // --- Event Replay ---
        const PLAYER_SWITCH_DELAY_MS = 2000; // 每切换一个玩家延迟 2 秒

        /** 仅对“出牌回合”（摸牌/打牌）返回 actor，用于按玩家分批；定缺等不拆批 */
        function eventActor(ev) {
            if (ev.type === 'tsumo' || ev.type === 'dahai') return ev.actor;
            return null;
        }
        /** 按玩家出牌回合拆批：同一玩家的 tsumo+dahai 为一批，换玩家时新一批，每批之间延迟 2 秒 */
        function splitEventsByPlayerTurn(events) {
            const batches = [];
            let cur = [];
            let lastActor = null;
            for (const ev of events) {
                const actor = eventActor(ev);
                if (actor !== undefined && actor !== null && lastActor !== null && actor !== lastActor) {
                    batches.push(cur);
                    cur = [];
                }
                if (actor !== undefined && actor !== null) lastActor = actor;
                cur.push(ev);
            }
            if (cur.length) batches.push(cur);
            return batches;
        }

        function applyOneEvent(ev) {
            switch (ev.type) {
                case 'start_game':
                    state.gaming = true;
                    state.gameEnded = false;
                    state.scores = [60000, 60000, 60000, 60000];
                    state.fuuro = [[], [], [], []];
                    break;
                case 'start_kyoku':
                    state.gaming = true;
                    state.gameEnded = false;
                    state.scores = (ev.scores && ev.scores.length === 4) ? [...ev.scores] : [60000, 60000, 60000, 60000];
                    state.tehai = (ev.tehais ? ev.tehais[state.myPlayerId] : []) || [];
                    sortTiles(state.tehai);
                    state.tsumoTile = null;
                    state.discards = [[], [], [], []];
                    state.agari = [false, false, false, false];
                    state.dingque = [null, null, null, null];
                    state.fuuro = [[], [], [], []];
                    state.tilesLeft = 56;
                    break;
                case 'ding_que':
                    if (ev.actor !== undefined && (ev.suit || ev.color)) {
                        state.dingque[ev.actor] = ev.suit || ev.color;
                    }
                    break;
                case 'tsumo':
                    state.currentActor = ev.actor;
                    state.tilesLeft = Math.max(0, state.tilesLeft - 1);
                    if (ev.actor === state.myPlayerId && ev.pai && ev.pai !== '?') {
                        state.tsumoTile = ev.pai;
                    }
                    break;
                case 'dahai':
                    state.currentActor = (ev.actor + 1) % 4;
                    if (!state.discards[ev.actor]) state.discards[ev.actor] = [];
                    state.discards[ev.actor].push(ev.pai);
                    if (ev.actor === state.myPlayerId) {
                        if (state.tsumoTile === ev.pai) {
                            state.tsumoTile = null;
                        } else {
                            const idx = state.tehai.indexOf(ev.pai);
                            if (idx > -1) state.tehai.splice(idx, 1);
                            if (state.tsumoTile) {
                                state.tehai.push(state.tsumoTile);
                                sortTiles(state.tehai);
                                state.tsumoTile = null;
                            }
                        }
                    }
                    ui.selectedIdx = -1;
                    break;
                case 'pon':
                case 'kan':
                case 'ankan':
                case 'daiminkan':
                case 'kakan':
                    state.currentActor = ev.actor;
                    if (ev.actor === state.myPlayerId) {
                        let toRemove = ev.type === 'kakan' && ev.pai ? [ev.pai] : (ev.consumed || []);
                        for (const t of toRemove) {
                            if (state.tsumoTile === t) state.tsumoTile = null;
                            else {
                                const idx = state.tehai.indexOf(t);
                                if (idx > -1) state.tehai.splice(idx, 1);
                            }
                        }
                        if (state.tsumoTile) {
                            state.tehai.push(state.tsumoTile);
                            state.tsumoTile = null;
                        }
                        sortTiles(state.tehai);
                    }
                    if (!state.fuuro[ev.actor]) state.fuuro[ev.actor] = [];
                    if (ev.type === 'kakan') {
                        const p = ev.pai;
                        const target = state.fuuro[ev.actor].find(m =>
                            m.type === 'pon' && m.tiles[0] && m.tiles[0][0] == p[0] && m.tiles[0][1] == p[1]
                        );
                        if (target) { target.type = 'kakan'; target.tiles.push(p); }
                        else state.fuuro[ev.actor].push({ type: 'kakan', tiles: [p, p, p, p] });
                    } else {
                        const tiles = ev.type === 'ankan' ? ev.consumed : [ev.pai, ...(ev.consumed || [])];
                        state.fuuro[ev.actor].push({ type: ev.type, tiles });
                    }
                    break;
                case 'agari':
                case 'hora':
                    state.agari[ev.actor] = true;
                    break;
            }
        }

        function applyBatch(batch) {
            state.tsumoTile = null;
            let lastAiDiscardPai = null;
            for (const ev of batch) {
                applyOneEvent(ev);
                if (ev.type === 'dahai' && ev.actor !== state.myPlayerId && ev.pai && ev.pai !== '?') {
                    lastAiDiscardPai = ev.pai;
                }
            }
            if (lastAiDiscardPai) playTileSound(lastAiDiscardPai);
        }

        function replayEvents(events) {
            const batches = splitEventsByPlayerTurn(events);
            if (batches.length === 0) return;
            applyBatch(batches[0]);
            for (let i = 1; i < batches.length; i++) {
                setTimeout(() => applyBatch(batches[i]), PLAYER_SWITCH_DELAY_MS * i);
            }
        }

        // --- Interaction ---
        function startGame() {
            send({ type: 'start_game' });
        }

        function doDingQue(suit) {
            send({ type: 'ding_que', suit: suit });
            const map = { 'm': 'man', 'p': 'pin', 's': 'sou' };
            state.dingque[state.myPlayerId] = map[suit] || suit;
            state.phase = 'playing';
        }

        /** 定缺花色字符（m/p/s），未定缺为 null */
        function getDingqueSuitChar() {
            const dq = state.dingque[state.myPlayerId];
            if (!dq) return null;
            const map = { man: 'm', pin: 'p', sou: 's', m: 'm', p: 'p', s: 's' };
            return map[dq] || dq;
        }
        /** 手牌（含自摸牌）是否还有定缺花色的牌 */
        function hasDingqueTilesInHand() {
            const suit = getDingqueSuitChar();
            if (!suit) return false;
            const handTiles = [...state.tehai];
            if (state.tsumoTile) handTiles.push(state.tsumoTile);
            return handTiles.some(t => t && t[1] === suit);
        }
        /** 当前是否允许打出这张牌（定缺未打完时只能打定缺花色） */
        function canDiscardTile(tile) {
            if (!tile || tile === '?') return false;
            const suit = getDingqueSuitChar();
            if (!suit) return true;
            if (!hasDingqueTilesInHand()) return true;
            return tile[1] === suit;
        }

        function onTileClick(tile, idx) {
            if (!state.canDiscard) return;
            if (tile === '?') return;
            if (!canDiscardTile(tile)) return; // 定缺未打完时只能打定缺牌

            if (ui.selectedIdx === idx) {
                send({ type: 'dahai', actor: state.myPlayerId, pai: tile });
                state.tsumoTile = null;
                state.canDiscard = false;
                ui.selectedIdx = -1;
            } else {
                ui.selectedIdx = idx;
            }
        }

        function doAction(act) {
            send({ type: 'action', action: act });
            state.validActions = [];
            state.canDiscard = false;
        }

        function actionLabel(type) {
            const map = { 'hu': '胡', 'pon': '碰', 'kan': '杠', 'pass': '过' };
            return map[type] || type.toUpperCase();
        }

        function dingqueLabel(suit) {
            const map = { 'm': '万', 'p': '筒', 's': '条', 'man': '万', 'pin': '筒', 'sou': '条' };
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
            if (offset === 0) return state.tehai;
            return [];
        }

        function discards(offset) {
            const id = (state.myPlayerId + offset) % 4;
            return state.discards[id] || [];
        }

        function getFuuro(offset) {
            const id = (state.myPlayerId + offset) % 4;
            return state.fuuro[id] || [];
        }

        function isRecommended(tile) {
            return analysis.best_action && analysis.best_action.pai === tile;
        }

        /** AI 玩家河牌中是否为“刚打出”的那张（用于播放动画） */
        function isLastDiscard(offset, index) {
            if (offset === 0) return false; // 自己的河牌不播动画
            if (!state.gaming || state.gameEnded) return false;
            const d = discards(offset);
            return d.length > 0 && index === d.length - 1;
        }

        // Init
        connect();

        return {
            state, analysis, ui, isMyTurn,
            startGame, doDingQue, onTileClick, doAction,
            tileSrc, player, hand, discards, isRecommended, actionLabel, dingqueLabel,
            getFuuro, canDiscardTile, isLastDiscard,
            getHandTileCount(p) {
                const meldCount = state.fuuro[p] ? state.fuuro[p].length : 0;
                let count = 13 - (meldCount * 3);
                // If it's AI's turn, they have one extra tile (drawn)
                if (state.currentActor === p && state.gaming && !state.gameEnded) {
                    count += 1;
                }
                return Math.max(0, count);
            }
        };
    }
});

app.mount('#app');
