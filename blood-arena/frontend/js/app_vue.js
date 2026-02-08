const { createApp, reactive, computed, onMounted } = Vue;

/* --- Constants & Mappers --- */
const TSUPAIS = [null, "E", "S", "W", "N", "P", "F", "C"]; // 1-7
const TSUPAI_TO_IMG = {
    "E": "ji_e", "S": "ji_s", "W": "ji_w", "N": "ji_n",
    "P": "no", "F": "ji_h", "C": "ji_c",
    "1z": "ji_e", "2z": "ji_s", "3z": "ji_w", "4z": "ji_n",
    "5z": "no", "6z": "ji_h", "7z": "ji_c"
};

function getPaiImage(pai, pose = 0) {
    if (!pai || pai === "?") return `/static/images/p_bk_${pose}.gif`;

    // Parse: 5m, 5mr, E, 1z
    let name = "";
    let ext = "gif";

    // Check Honor (z or Letter)
    if (TSUPAI_TO_IMG[pai] || /^[1-7]z$/.test(pai)) {
        name = TSUPAI_TO_IMG[pai] || TSUPAI_TO_IMG[pai.replace("z", "")]; // Fallback
        if (!name && /^[1-7]z$/.test(pai)) {
            // map 1z->E->ji_e
            const idx = parseInt(pai[0]);
            const letter = TSUPAIS[idx];
            name = TSUPAI_TO_IMG[letter];
        }
    } else {
        // Suits: 1m, 5p, 9s
        const match = pai.match(/^([0-9])([mps])(r)?$/);
        if (match) {
            const num = match[1];
            const type = match[2];
            const red = match[3];
            name = `${type}s${num}${red ? "r" : ""}`;
            if (red) ext = "png";
        } else {
            // Fallback for unexpected formats
            // console.warn("Unknown pai format:", pai);
            return `/static/images/p_bk_${pose}.gif`;
        }
    }

    return `/static/images/p_${name}_${pose}.${ext}`;
}

const app = createApp({
    setup() {
        // --- State ---
        const state = reactive({
            connected: false,
            status: '连接中…',
            notification: "", // Big center text

            // Game Data
            players: [
                { name: '我', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null },   // 0
                { name: '对手1', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null }, // 1
                { name: '对手2', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null }, // 2
                { name: '对手3', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null }  // 3
            ],

            // Turn Logic
            myPlayerId: 0,
            currentActor: -1,
            isMyTurn: false,
            validActions: [],

            // Meta（血战到底：无宝牌，可选剩余牌数）
            gameStarted: false,
            tilesLeft: null
        });

        let ws = null;

        // --- Helpers ---
        const playSound = (type) => {
            // TODO: Add Audio 
        };

        const sortHand = (tehai) => {
            // Simple sort: m < p < s < z
            const weight = (p) => {
                if (p === "?") return 999;
                if (TSUPAI_TO_IMG[p] || /^[1-7]z$/.test(p)) {
                    // Honor
                    let idx = 0;
                    if (p.endsWith('z')) idx = parseInt(p[0]);
                    else idx = TSUPAIS.indexOf(p);
                    return 300 + idx;
                }
                const match = p.match(/^([0-9])([mps])/);
                if (!match) return 999;
                const n = parseInt(match[1]);
                const t = match[2];
                let base = 0;
                if (t === 'm') base = 0;
                if (t === 'p') base = 100;
                if (t === 's') base = 200;
                return base + n;
            };
            return tehai.sort((a, b) => weight(a) - weight(b));
        };

        // --- Core Logic ---

        // Evaluates the current state to determine what UI to show (Phase Detection)
        const evaluatePhase = (events) => {
            // 1. Ding Que Phase Detection
            // Condition: StartKyoku exists events BUT NO one has discarded or melded yet?
            // Or simpler: My DingQueSuit is null?
            const lastStartKyokuIdx = events.findLastIndex(e => e.type === 'start_kyoku');

            if (lastStartKyokuIdx !== -1) {
                // Check if any DingQue action happened AFTER start_kyoku for ME
                // Actually, the server broadcasts 'ding_que' event when SOMEONE does it.
                // We need to know if *I* have done it.
                // state.players[0].dingQueSuit reflects the 'ding_que' event.
                // But strictly, we should offer the buttons if we haven't sent it yet.
                // Best way: Check if any play events (Dahai/Chi/Pon/Tsumo) happened after Start Kyoku.
                // UPDATE: 'tsumo' is allowed (Dealer draws 14th tile BEFORE Ding Que).
                // So we only look for Discards or Melds (Dahai, Chi, Pon, Kan).

                const eventsAfterKyoku = events.slice(lastStartKyokuIdx + 1);
                const hasPlay = eventsAfterKyoku.some(e => ['dahai', 'chi', 'pon', 'daiminkan', 'kan'].includes(e.type));

                if (!hasPlay) {
                    // We are in Ding Que or Pre-Game phase
                    if (!state.players[state.myPlayerId].dingQueSuit) {
                        // I haven't picked a suit yet -> Show Buttons
                        state.status = "Select Void Suit (Ding Que)";
                        state.validActions = [
                            { label: '万', class: 'btn-action btn-man', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'man' } },
                            { label: '筒', class: 'btn-action btn-pin', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'pin' } },
                            { label: '条', class: 'btn-action btn-sou', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'sou' } }
                        ];
                        return; // Exclusive UI
                    } else {
                        // I have picked.
                        if (state.isMyTurn) {
                            state.status = "YOUR TURN (Discard)";
                            state.validActions = []; // Clear buttons, allow tile click
                        } else {
                            state.status = "Waiting for other players to Ding Que...";
                            state.validActions = [];
                        }
                        // Don't return, maybe we want to render the board? Yes.
                    }
                }
            }

            // 2. My Turn Detection (Playing Phase)
            // Already handled by detailed event processing (Tsumo/Pon logic),
            // but we can sanity check here.
            // If validActions is empty and it's my turn (from Tsumo), ensure I can Dahai?
            // Handled by handleTileClick.
        };

        const handleEvent = (event) => {
            // console.log("Event:", event);
            const { type, actor, target, pai, consumed, scores, tehais, suit } = event;

            if (type === 'start_game') {
                state.gameStarted = true;
                state.status = "Game Started";
                state.players.forEach(p => {
                    p.tehai = []; p.discards = []; p.melds = []; p.score = 60000; p.dingQueSuit = null;
                });
                state.tilesLeft = null;
            }
            else if (type === 'start_kyoku') {
                state.gameStarted = true;
                state.status = '对局开始';
                if (scores) scores.forEach((s, i) => state.players[i].score = s);
                state.players.forEach(p => p.dingQueSuit = null); // Reset Ding Que

                if (tehais) {
                    tehais.forEach((hand, i) => {
                        state.players[i].tehai = [...hand];
                        if (i === state.myPlayerId) sortHand(state.players[i].tehai);
                    });
                }
            }
            else if (type === 'tsumo') {
                state.status = `玩家${actor} 摸牌`;
                state.currentActor = actor;
                state.players[actor].tehai.push(pai);
                if (actor === state.myPlayerId) {
                    state.isMyTurn = true;
                    state.status = '轮到你出牌';
                    state.validActions = [
                        { label: '自摸', type: 'hora', payload: { type: 'hora', actor: state.myPlayerId, target: state.myPlayerId }, class: 'btn-action btn-win' }
                    ];
                }
            }
            else if (type === 'dahai') {
                state.status = `玩家${actor} 出牌`;
                state.currentActor = actor;
                const pIdx = state.players[actor].tehai.indexOf(pai);
                if (pIdx !== -1) state.players[actor].tehai.splice(pIdx, 1);
                else if (actor !== state.myPlayerId) state.players[actor].tehai.pop();

                state.players[actor].discards.push(pai);
                if (actor === state.myPlayerId) sortHand(state.players[actor].tehai);

                if (actor !== state.myPlayerId && state.players[state.myPlayerId].dingQueSuit !== null) {
                    state.validActions = [
                        { label: '荣和', type: 'hora', payload: { type: 'hora', actor: state.myPlayerId, target: actor }, class: 'btn-action btn-win' },
                        { label: '碰', type: 'pon', payload: { type: 'pon', actor: state.myPlayerId, target: actor, pai: pai, consumed: [] }, class: 'btn-action' },
                        { label: '杠', type: 'daiminkan', payload: { type: 'daiminkan', actor: state.myPlayerId, target: actor, pai: pai }, class: 'btn-action' },
                        { label: '过', type: 'none', payload: { type: 'none' }, class: 'btn-action' }
                    ];
                }
            }
            else if (type === 'pon' || type === 'daiminkan' || type === 'chi') {
                // Remove consumed
                consumed.forEach(c => {
                    const idx = state.players[actor].tehai.indexOf(c);
                    if (idx !== -1) state.players[actor].tehai.splice(idx, 1);
                    else state.players[actor].tehai.pop();
                });
                state.players[actor].melds.push({ type, pai, consumed });
                // Remove from discards
                const targetDiscards = state.players[target].discards;
                if (targetDiscards.length > 0) targetDiscards.pop();

                state.currentActor = actor;
                if (actor === state.myPlayerId) {
                    state.isMyTurn = true;
                    state.validActions = []; // Must discard
                }
            }
            else if (type === 'ding_que') {
                state.status = `玩家${actor} 定缺`;
                state.players[actor].dingQueSuit = suit;
            }
            else if (type === 'hora') {
                state.notification = `和牌！玩家${actor}`;
                if (event.deltas && event.deltas.length === 4) {
                    event.deltas.forEach((d, i) => state.players[i].score += d);
                }
                setTimeout(() => state.notification = "", 5000);
            }
            else if (type === 'game_over') {
                state.notification = '对局结束';
            }
        };

        // --- Interaction ---
        const handleTileClick = (tile, index) => {
            if (state.isMyTurn) {
                // Determine if valid discard (e.g. Ding Que suit check)
                // For MVP, just send it.
                ws.send(JSON.stringify({
                    type: "dahai",
                    actor: state.myPlayerId,
                    pai: tile,
                    tsumogiri: false
                }));
                state.isMyTurn = false;
                state.validActions = []; // Clear buttons
            }
        };

        const sendAction = (action) => {
            ws.send(JSON.stringify(action.payload));
            state.validActions = [];
        };

        const tryStartGame = () => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                console.log("Sending manual start_game...");
                ws.send(JSON.stringify({ type: "start_game" }));
                state.status = "Start Requested...";
            } else {
                window.location.reload();
            }
        };

        // --- Setup ---
        onMounted(() => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);
            ws.onopen = () => {
                state.connected = true;
                // Removed initial start_game, now manual via tryStartGame or backend push
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === "state_update") {
                    // Full Replay to ensure sync
                    // Reset internal state first
                    state.players.forEach(p => {
                        p.tehai = []; p.discards = []; p.melds = []; p.dingQueSuit = null;
                    });
                    state.tilesLeft = null;

                    // Replay all
                    msg.data.events.forEach(handleEvent);

                    // Post-Process Logic (Phase Detection)
                    evaluatePhase(msg.data.events);
                }
            };
            ws.onclose = () => state.connected = false;
        });

        const suitName = (suit) => {
            if (!suit) return '';
            return { man: '万', pin: '筒', sou: '条' }[suit] || suit;
        };

        return {
            state,
            getPaiImage,
            suitName,
            handleTileClick,
            sendAction,
            tryStartGame
        };
    }
});

app.mount("#app");
