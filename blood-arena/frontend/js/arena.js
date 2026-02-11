/**
 * Blood Arena Frontend Logic
 * Supports: Phase Management, Action Buttons, AI Suggestions, Melds (Fuuro)
 */
import { createApp, reactive, computed } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';

const TILE_BASE = '/static/images/tiles';
function tileSrc(tile) {
    if (!tile || tile === 'back' || tile === '?') return `${TILE_BASE}/Back.png`;
    const n = tile[0], s = tile[1];
    const suit = { m: 'Man', p: 'Pin', s: 'Sou' }[s] || 'Man';
    return `${TILE_BASE}/${suit}${n}.png`;
}

function playDiscardSound(pai) {
    if (!pai || pai === '?') return;
    const src = `/static/audio/${pai}.m4a`;
    try {
        const a = new Audio(src);
        a.volume = 0.5;
        a.play().catch(() => {});
    } catch (_) {}
}

function playActionSound(filename) {
    if (!filename) return;
    const src = `/static/audio/${filename}`;
    try {
        const a = new Audio(src);
        a.volume = 0.6;
        a.play().catch(() => {});
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
            scores: [60000, 60000, 60000, 60000],
            tehai: [], // My hand
            tsumoTile: null, // Last drawn tile
            discards: [[], [], [], []],
            agari: [false, false, false, false],
            dingque: [null, null, null, null], // m, p, s
            fuuro: [[], [], [], []], // Melds

            // 8 局制：当前局数 1..8，累计输赢（相对每局 60000），是否整场已结束
            gameNumber: 0,
            matchDeltas: [0, 0, 0, 0],
            matchOver: false,

            // 已重放的事件数，用于只对“新动作”做延迟，避免每次从头重放
            lastReplayedEventCount: 0,

            // Interactive
            validActions: [],
            canDiscard: false,
            validActionsShown: false, // 碰/杠/胡：仅在与对手出牌同步后显示
            optimisticDahai: null, // 刚乐观更新打出的牌，replay 时跳过避免一对牌被移除两次

            // 是否在牌局结束时显示 AI 实际牌面（需要后端 game_over 带 tehais）
            showAiHands: true,
            finalTehais: null, // 牌局结束时四家手牌 [p0[], p1[], p2[], p3[]]
        });

        const ui = reactive({ selectedIdx: -1 });
        let ws = null;

        // --- Computed ---
        const isMyTurn = computed(() => state.currentActor === state.myPlayerId);
        const turnIndicatorText = computed(() => {
            if (!state.gaming || state.gameEnded) return '';
            if (state.currentActor === state.myPlayerId) return '你的回合';
            if (state.currentActor === -1) return '等待中...';
            const names = ['下家', '对家', '上家'];
            const idx = state.currentActor >= 1 && state.currentActor <= 3 ? state.currentActor - 1 : 0;
            return `等待 ${names[idx]}...`;
        });

        // --- Network ---
        const connect = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);

            ws.onopen = () => { state.connected = true; };
            ws.onclose = () => {
                state.connected = false;
                setTimeout(connect, 3000);
            };
            let msgQueue = [];
            let processing = false;
            ws.onmessage = (e) => {
                msgQueue.push(JSON.parse(e.data));
                if (!processing) {
                    processing = true;
                    (async function drain() {
                        while (msgQueue.length > 0) {
                            await handleMessage(msgQueue.shift());
                        }
                        processing = false;
                    })();
                }
            };
        };

        const send = (data) => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
        };

        // --- Message Handling ---
        async function handleMessage(msg) {
            if (msg.type === 'state_update') {
                await updateFullState(msg.data);
            } else if (msg.type === 'action_request') {
                handleActionRequest(msg.actions);
            } else if (msg.type === 'game_over') {
                if (msg.scores && msg.scores.length === 4) {
                    state.scores = [...msg.scores];
                }
                if (msg.match_deltas && msg.match_deltas.length === 4) {
                    state.matchDeltas = [...msg.match_deltas];
                }
                state.gameNumber = msg.game_number || 0;
                if (msg.tehais && msg.tehais.length === 4) {
                    state.finalTehais = msg.tehais.map(h => [...(h || [])]);
                } else {
                    state.finalTehais = null;
                }
                if (msg.is_match_over) {
                    state.matchOver = true;
                    state.gameEnded = true;
                    state.phase = 'result';
                    state.gaming = false;
                }
                state.canDiscard = false;
                state.validActions = [];
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

            const hasPonKanHu = state.validActions.some(a => ['pon', 'kan', 'hu'].includes(a.type));
            state.validActionsShown = !hasPonKanHu;

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

        let replayId = 0;
        async function updateFullState(data) {
            if (!data) return;
            const hasAuthoritativeHand = !!(data.tehais && Array.isArray(data.tehais) && data.tehais[state.myPlayerId]);
            if (hasAuthoritativeHand) {
                state.tehai = [...(data.tehais[state.myPlayerId] || [])];
                sortTiles(state.tehai);
                state.tsumoTile = data.my_tsumo ?? null;
                state.optimisticDahai = null;  // 权威状态覆盖，无需乐观更新标记
            }
            if (data.events) {
                replayId += 1;
                void replayEvents(data.events, replayId, hasAuthoritativeHand);
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

        const ACTION_DELAY_MS = 1500;
        const delay = (ms) => new Promise((r) => setTimeout(r, ms));

        // --- Event Replay ---
        // 增量重放：仅处理新增事件；fire-and-forget 以便 action_request 能立即处理，用户可出牌
        // hasAuthoritativeHand: 当 state_update 含 tehais 时，手牌由后端权威状态覆盖，replay 不修改 tehai/tsumoTile
        async function replayEvents(events, myReplayId, hasAuthoritativeHand = false) {
            if (!events || events.length === 0) return;
            const last = state.lastReplayedEventCount;
            if (events.length === last) return;
            if (myReplayId !== undefined && myReplayId !== replayId) return; // 被更新的 replay 取代
            const startIdx = (events.length < last) ? 0 : last; // 新局从头，否则只处理新增
            if (startIdx === 0 && !hasAuthoritativeHand) state.tsumoTile = null;

            const isAction = (ev) => ['dahai', 'pon', 'kan', 'ankan', 'daiminkan', 'kakan'].includes(ev.type);
            let firstNewAction = true;
            for (let i = startIdx; i < events.length; i++) {
                const ev = events[i];
                if (isAction(ev)) {
                    if (!firstNewAction) {
                        await delay(ACTION_DELAY_MS);
                        if (myReplayId !== undefined && myReplayId !== replayId) return;
                    }
                    firstNewAction = false;
                }
                switch (ev.type) {
                    case 'start_game':
                        state.gaming = true;
                        state.gameEnded = false;
                        state.matchOver = false;
                        state.gameNumber = 0;
                        state.matchDeltas = [0, 0, 0, 0];
                        state.lastReplayedEventCount = 0;
                        state.finalTehais = null;
                        state.scores = [60000, 60000, 60000, 60000];
                        state.fuuro = [[], [], [], []];
                        state.optimisticDahai = null;
                        state.canDiscard = false;
                        state.validActions = [];
                        state.validActionsShown = false;
                        break;
                    case 'start_kyoku':
                        state.gaming = true;
                        if (!state.matchOver) state.gameEnded = false;
                        state.lastReplayedEventCount = 0;
                        state.finalTehais = null;
                        state.scores = (ev.scores && ev.scores.length === 4) ? [...ev.scores] : [60000, 60000, 60000, 60000];
                        if (!hasAuthoritativeHand) {
                            state.tehai = (ev.tehais ? ev.tehais[state.myPlayerId] : []) || [];
                            sortTiles(state.tehai);
                            state.tsumoTile = null;
                        }
                        state.optimisticDahai = null;
                        state.discards = [[], [], [], []];
                        state.agari = [false, false, false, false];
                        state.dingque = [null, null, null, null];
                        state.fuuro = [[], [], [], []];
                        state.tilesLeft = 56;
                        state.canDiscard = false;
                        state.validActions = [];
                        state.validActionsShown = false;
                        break;
                    case 'ding_que':
                        if (ev.actor !== undefined && (ev.suit || ev.color)) {
                            state.dingque[ev.actor] = ev.suit || ev.color;
                        }
                        break;
                    case 'tsumo':
                        state.currentActor = ev.actor;
                        state.tilesLeft = Math.max(0, state.tilesLeft - 1);
                        if (!hasAuthoritativeHand && ev.actor === state.myPlayerId && ev.pai && ev.pai !== '?') {
                            if (state.tehai.length < 14) state.tsumoTile = ev.pai;
                        }
                        break;
                    case 'dahai':
                        if (ev.actor !== state.myPlayerId) {
                            state.validActionsShown = true;
                            playDiscardSound(ev.pai);
                        }
                        let nextActor = (ev.actor + 1) % 4;
                        for (let _ = 0; _ < 4; _++) {
                            if (!state.agari[nextActor]) break;
                            nextActor = (nextActor + 1) % 4;
                        }
                        state.currentActor = nextActor;
                        if (!state.discards[ev.actor]) state.discards[ev.actor] = [];
                        const expectedDiscardCount = events.slice(0, i + 1).filter(e => e.type === 'dahai' && e.actor === ev.actor).length;
                        if (state.discards[ev.actor].length < expectedDiscardCount) {
                            state.discards[ev.actor].push(ev.pai);
                        }
                        if (!hasAuthoritativeHand && ev.actor === state.myPlayerId) {
                            if (state.tsumoTile === ev.pai) {
                                state.tsumoTile = null;
                                state.optimisticDahai = null;
                            } else if (state.optimisticDahai === ev.pai) {
                                state.tsumoTile = null;
                            } else {
                                const idx = state.tehai.indexOf(ev.pai);
                                if (idx > -1) {
                                    state.tehai.splice(idx, 1);
                                    if (state.tsumoTile) {
                                        state.tehai.push(state.tsumoTile);
                                        sortTiles(state.tehai);
                                        state.tsumoTile = null;
                                    }
                                    state.optimisticDahai = null;
                                } else {
                                    state.tsumoTile = null;
                                    state.optimisticDahai = null;
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
                        if (ev.type === 'pon') playActionSound('pon.m4a');
                        else playActionSound('kan.m4a');
                        if (!hasAuthoritativeHand && ev.actor === state.myPlayerId) {
                            let toRemove = [];
                            if (ev.type === 'kakan') {
                                if (ev.pai) toRemove.push(ev.pai);
                            } else {
                                if (ev.consumed) toRemove = [...ev.consumed];
                            }
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
                            if (target) {
                                target.type = 'kakan';
                                target.tiles.push(p);
                            } else {
                                state.fuuro[ev.actor].push({ type: 'kakan', tiles: [p, p, p, p] });
                            }
                        } else {
                            let tiles = [];
                            if (ev.type === 'ankan') tiles = ev.consumed;
                            else if (ev.type === 'daiminkan') tiles = [ev.pai, ...(ev.consumed || [])];
                            else if (ev.type === 'pon') tiles = [ev.pai, ...(ev.consumed || [])];
                            else tiles = [ev.pai, ...(ev.consumed || [])];
                            state.fuuro[ev.actor].push({ type: ev.type, tiles });
                        }
                        break;
                    case 'agari':
                    case 'hora':
                        state.agari[ev.actor] = true;
                        playActionSound((ev.target === undefined || ev.actor === ev.target) ? 'tsumo.m4a' : 'ron.m4a');
                        break;
                }
            }
            state.lastReplayedEventCount = events.length;
        }

        // --- Interaction ---
        function startGame() {
            state.gameNumber = 0;
            state.matchDeltas = [0, 0, 0, 0];
            state.matchOver = false;
            state.gameEnded = false;
            state.lastReplayedEventCount = 0;
            send({ type: 'start_game' });
        }

        /** 当前显示的局数（1..8），对局中为正在打的局，结束后为 8 */
        function currentGameDisplay() {
            if (state.matchOver) return 8;
            return state.gaming ? (state.gameNumber + 1) : Math.max(1, state.gameNumber);
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
            if (state.currentActor !== state.myPlayerId) return;
            if (tile === '?') return;
            if (!canDiscardTile(tile)) return; // 定缺未打完时只能打定缺牌

            if (ui.selectedIdx === idx) {
                playDiscardSound(tile);
                send({ type: 'dahai', actor: state.myPlayerId, pai: tile });
                if (idx === 'tsumo') {
                    state.tsumoTile = null;
                } else {
                    const i = state.tehai.indexOf(tile);
                    if (i > -1) {
                        state.tehai.splice(i, 1);
                        if (state.tsumoTile) {
                            state.tehai.push(state.tsumoTile);
                            sortTiles(state.tehai);
                        }
                        state.optimisticDahai = tile;
                    }
                    state.tsumoTile = null;
                }
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

        /** 用于显示的手牌：牌局结束且有 finalTehais 时返回该玩家牌面，否则自家为 tehai，他家为空 */
        function handTiles(offset) {
            const id = (state.myPlayerId + offset) % 4;
            if (state.gameEnded && state.finalTehais && state.finalTehais[id]) {
                return state.finalTehais[id];
            }
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

        // Init
        connect();

        /** 累计输赢文案：自家/下家/对家/上家 */
        function matchDeltaLabel(offset) {
            const id = (state.myPlayerId + offset) % 4;
            const d = state.matchDeltas[id];
            if (d == null) return '';
            const sign = d >= 0 ? '+' : '';
            return sign + d;
        }

        return {
            state, ui, isMyTurn, turnIndicatorText,
            startGame, doDingQue, onTileClick, doAction,
            tileSrc, player, hand, discards, actionLabel, dingqueLabel,
            getFuuro, canDiscardTile, handTiles, currentGameDisplay, matchDeltaLabel,
            getHandTileCount(p) {
                const fuuro = state.fuuro[p] || [];
                const meldTiles = fuuro.reduce((sum, m) => sum + (m.tiles ? m.tiles.length : 3), 0);
                let count = 13 - meldTiles;
                if (state.currentActor === p && state.gaming && !state.gameEnded) {
                    count += 1;
                }
                return Math.max(0, count);
            }
        };
    }
});

app.mount('#app');
