const { createApp, reactive, computed, onMounted } = Vue;

const TSUPAIS = [null, "E", "S", "W", "N", "P", "F", "C"];
const TSUPAI_TO_IMG = {
    "E": "ji_e", "S": "ji_s", "W": "ji_w", "N": "ji_n",
    "P": "no", "F": "ji_h", "C": "ji_c",
    "1z": "ji_e", "2z": "ji_s", "3z": "ji_w", "4z": "ji_n",
    "5z": "no", "6z": "ji_h", "7z": "ji_c"
};

function getPaiImage(pai, pose = 0) {
    if (!pai || pai === "?") return `/static/images/p_bk_${pose}.gif`;
    let name = "";
    let ext = "gif";
    if (TSUPAI_TO_IMG[pai] || /^[1-7]z$/.test(pai)) {
        name = TSUPAI_TO_IMG[pai] || TSUPAI_TO_IMG[pai.replace("z", "")];
        if (!name && /^[1-7]z$/.test(pai)) {
            const idx = parseInt(pai[0]);
            const letter = TSUPAIS[idx];
            name = TSUPAI_TO_IMG[letter];
        }
    } else {
        const match = pai.match(/^([0-9])([mps])(r)?$/);
        if (match) {
            const num = match[1];
            const type = match[2];
            const red = match[3];
            name = `${type}s${num}${red ? "r" : ""}`;
            if (red) ext = "png";
        } else {
            return `/static/images/p_bk_${pose}.gif`;
        }
    }
    return `/static/images/p_${name}_${pose}.${ext}`;
}

const app = createApp({
    setup() {
        const state = reactive({
            connected: false,
            status: '连接中…',
            notification: "",
            players: [
                { name: '我', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null },
                { name: '对手1', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null },
                { name: '对手2', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null },
                { name: '对手3', score: 60000, tehai: [], discards: [], melds: [], dingQueSuit: null }
            ],
            myPlayerId: 0,
            currentActor: -1,
            isMyTurn: false,
            validActions: [],
            gameStarted: false,
            tilesLeft: null,
            gameEnded: false,
            debug: false
        });

        let ws = null;

        const sortHand = (tehai) => {
            const weight = (p) => {
                if (p === "?") return 999;
                if (TSUPAI_TO_IMG[p] || /^[1-7]z$/.test(p)) {
                    let idx = p.endsWith('z') ? parseInt(p[0]) : TSUPAIS.indexOf(p);
                    return 300 + idx;
                }
                const match = p.match(/^([0-9])([mps])/);
                if (!match) return 999;
                const n = parseInt(match[1]);
                const t = match[2];
                const base = t === 'm' ? 0 : t === 'p' ? 100 : 200;
                return base + n;
            };
            return tehai.sort((a, b) => weight(a) - weight(b));
        };

        const evaluatePhase = (events) => {
            const lastStartKyokuIdx = events.findLastIndex(e => e.type === 'start_kyoku');
            if (lastStartKyokuIdx !== -1) {
                const eventsAfterKyoku = events.slice(lastStartKyokuIdx + 1);
                const hasPlay = eventsAfterKyoku.some(e => ['dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(e.type));
                if (!hasPlay) {
                    if (!state.players[state.myPlayerId].dingQueSuit) {
                        state.status = '请选择定缺花色';
                        state.validActions = [
                            { label: '万', class: 'btn-action btn-man', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'man' } },
                            { label: '筒', class: 'btn-action btn-pin', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'pin' } },
                            { label: '条', class: 'btn-action btn-sou', payload: { type: 'ding_que', actor: state.myPlayerId, suit: 'sou' } }
                        ];
                        return;
                    } else {
                        state.status = state.isMyTurn ? '轮到你出牌' : '等待其他玩家定缺…';
                        state.validActions = [];
                    }
                }
            }
        };

        const handleEvent = (event) => {
            const { type, actor, target, pai, consumed, scores, tehais, suit } = event;

            if (type === 'start_game') {
                state.gameStarted = true;
                state.gameEnded = false;
                state.status = '对局开始';
                state.players.forEach(p => {
                    p.tehai = []; p.discards = []; p.melds = []; p.score = 60000; p.dingQueSuit = null;
                });
                state.tilesLeft = null;
            }
            else if (type === 'start_kyoku') {
                state.gameStarted = true;
                state.status = '对局开始';
                if (scores) scores.forEach((s, i) => state.players[i].score = s);
                state.players.forEach(p => p.dingQueSuit = null);
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
                else state.players[actor].tehai.pop();
                state.players[actor].discards.push(pai);
                if (actor === state.myPlayerId) sortHand(state.players[actor].tehai);

                if (actor !== state.myPlayerId && state.players[state.myPlayerId].dingQueSuit !== null) {
                    const myHand = state.players[state.myPlayerId].tehai;
                    const ponConsumed = getPonConsumed(myHand, pai);
                    const minkanConsumed = getMinkanConsumed(myHand, pai);
                    const actions = [
                        { label: '荣和', payload: { type: 'hora', actor: state.myPlayerId, target: actor }, class: 'btn-action btn-win' },
                        { label: '过', payload: { type: 'none' }, class: 'btn-action' }
                    ];
                    if (ponConsumed.length >= 2)
                        actions.splice(1, 0, { label: '碰', payload: { type: 'pon', actor: state.myPlayerId, target: actor, pai: pai, consumed: ponConsumed }, class: 'btn-action' });
                    if (minkanConsumed.length >= 3)
                        actions.splice(1, 0, { label: '杠', payload: { type: 'daiminkan', actor: state.myPlayerId, target: actor, pai: pai, consumed: minkanConsumed }, class: 'btn-action' });
                    state.validActions = actions;
                }
            }
            else if (type === 'pon' || type === 'daiminkan' || type === 'chi' || type === 'ankan' || type === 'kakan') {
                const consumedList = consumed && Array.isArray(consumed) ? consumed : [];
                consumedList.forEach(c => {
                    const idx = state.players[actor].tehai.indexOf(c);
                    if (idx !== -1) state.players[actor].tehai.splice(idx, 1);
                    else state.players[actor].tehai.pop();
                });
                state.players[actor].melds.push({ type, pai: pai || (consumedList[0]), consumed: consumedList });
                if (target != null && state.players[target]) {
                    const targetDiscards = state.players[target].discards;
                    if (targetDiscards.length > 0) targetDiscards.pop();
                }
                state.currentActor = actor;
                if (actor === state.myPlayerId) {
                    state.isMyTurn = true;
                    state.validActions = [];
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
            else if (type === 'game_over' || type === 'end_game') {
                state.gameEnded = true;
                state.notification = '对局结束';
            }
        };

        const canSend = () => ws && ws.readyState === WebSocket.OPEN;

        const handleTileClick = (tile, index) => {
            if (!state.isMyTurn || !canSend()) return;
            ws.send(JSON.stringify({
                type: "dahai",
                actor: state.myPlayerId,
                pai: tile,
                tsumogiri: false
            }));
            state.isMyTurn = false;
            state.validActions = [];
        };

        const sendAction = (action) => {
            if (!canSend()) return;
            ws.send(JSON.stringify(action.payload));
            state.validActions = [];
        };

        const tryStartGame = () => {
            if (canSend()) {
                ws.send(JSON.stringify({ type: "start_game" }));
                state.status = '正在开局…';
                state.gameEnded = false;
            } else {
                state.status = '未连接，请刷新页面';
            }
        };

        onMounted(() => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);
            ws.onopen = () => {
                state.connected = true;
                if (!state.gameStarted) state.status = '已连接，请点击开始对局';
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === "state_update") {
                    state.gameEnded = false;
                    state.players.forEach(p => {
                        p.tehai = []; p.discards = []; p.melds = []; p.dingQueSuit = null;
                    });
                    state.tilesLeft = null;
                    msg.data.events.forEach(handleEvent);
                    evaluatePhase(msg.data.events);
                } else if (msg.type === "game_over") {
                    state.gameStarted = true;
                    state.gameEnded = true;
                    state.notification = '对局结束';
                    if (msg.scores && msg.scores.length === 4) {
                        msg.scores.forEach((s, i) => state.players[i].score = s);
                    }
                    state.validActions = [];
                    state.isMyTurn = false;
                }
            };
            ws.onclose = () => state.connected = false;
        });

        const suitName = (suit) => {
            if (!suit) return '';
            return { man: '万', pin: '筒', sou: '条' }[suit] || suit;
        };

        const playerZones = [
            { zone: 'top', seat: 2, pose: 2 },
            { zone: 'left', seat: 3, pose: 3 },
            { zone: 'right', seat: 1, pose: 1 },
            { zone: 'bottom', seat: 0, pose: 0 }
        ];

        const showResultPanel = computed(() => state.gameEnded);
        const isMyTurn = computed(() => state.isMyTurn);

        function getPonConsumed(tehai, pai) {
            if (!pai || !tehai || tehai.length < 2) return [];
            const out = [];
            for (let i = 0; i < tehai.length && out.length < 2; i++) {
                if (tehai[i] === pai) out.push(tehai[i]);
            }
            return out.length >= 2 ? out : [];
        }

        function getMinkanConsumed(tehai, pai) {
            if (!pai || !tehai || tehai.length < 3) return [];
            const out = [];
            for (let i = 0; i < tehai.length && out.length < 3; i++) {
                if (tehai[i] === pai) out.push(tehai[i]);
            }
            return out.length >= 3 ? out : [];
        }

        return {
            state,
            playerZones,
            showResultPanel,
            isMyTurn,
            getPaiImage,
            suitName,
            handleTileClick,
            sendAction,
            tryStartGame
        };
    }
});

app.mount("#app");
