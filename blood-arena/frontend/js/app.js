/**
 * 血战到底 - 人机对战
 * Vue 3 单页，WebSocket 与后端通信
 */
const { createApp, reactive, computed, onMounted } = Vue;

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
const TSUPAIS = [null, 'E', 'S', 'W', 'N', 'P', 'F', 'C'];
const TSUPAI_TO_IMG = {
    E: 'ji_e', S: 'ji_s', W: 'ji_w', N: 'ji_n', P: 'no', F: 'ji_h', C: 'ji_c',
    '1z': 'ji_e', '2z': 'ji_s', '3z': 'ji_w', '4z': 'ji_n',
    '5z': 'no', '6z': 'ji_h', '7z': 'ji_c'
};
const SUIT_NAMES = { man: '万', pin: '筒', sou: '条' };
const INIT_SCORE = 60000;
const PLAYER_ZONES = [
    { zone: 'top', seat: 2, pose: 2 },
    { zone: 'left', seat: 3, pose: 3 },
    { zone: 'right', seat: 1, pose: 1 },
    { zone: 'bottom', seat: 0, pose: 0 }
];

// ---------------------------------------------------------------------------
// 工具：牌面与手牌
// ---------------------------------------------------------------------------
function getPaiImage(pai, pose = 0) {
    if (!pai || pai === '?') return `/static/images/p_bk_${pose}.gif`;
    let name = '', ext = 'gif';
    if (TSUPAI_TO_IMG[pai] || /^[1-7]z$/.test(pai)) {
        name = TSUPAI_TO_IMG[pai] || TSUPAI_TO_IMG[pai.replace('z', '')];
        if (!name && /^[1-7]z$/.test(pai)) {
            const idx = parseInt(pai[0], 10);
            name = TSUPAI_TO_IMG[TSUPAIS[idx]];
        }
    } else {
        const m = pai.match(/^([0-9])([mps])(r)?$/);
        if (m) {
            name = `${m[2]}s${m[1]}${m[3] ? 'r' : ''}`;
            if (m[3]) ext = 'png';
        } else {
            return `/static/images/p_bk_${pose}.gif`;
        }
    }
    return `/static/images/p_${name}_${pose}.${ext}`;
}

function sortHand(tehai) {
    const weight = (p) => {
        if (p === '?') return 999;
        if (TSUPAI_TO_IMG[p] || /^[1-7]z$/.test(p)) {
            const idx = p.endsWith('z') ? parseInt(p[0], 10) : TSUPAIS.indexOf(p);
            return 300 + idx;
        }
        const m = p.match(/^([0-9])([mps])/);
        if (!m) return 999;
        const base = { m: 0, p: 100, s: 200 }[m[2]];
        return base + parseInt(m[1], 10);
    };
    return tehai.sort((a, b) => weight(a) - weight(b));
}

function getPonConsumed(tehai, pai) {
    if (!pai || !tehai?.length || tehai.length < 2) return [];
    const out = tehai.filter(t => t === pai).slice(0, 2);
    return out.length >= 2 ? out : [];
}

function getMinkanConsumed(tehai, pai) {
    if (!pai || !tehai?.length || tehai.length < 3) return [];
    const out = tehai.filter(t => t === pai).slice(0, 3);
    return out.length >= 3 ? out : [];
}

// ---------------------------------------------------------------------------
// 初始状态
// ---------------------------------------------------------------------------
function createInitialState() {
    const players = ['我', '对手1', '对手2', '对手3'].map(name => ({
        name,
        score: INIT_SCORE,
        tehai: [],
        discards: [],
        melds: [],
        dingQueSuit: null
    }));
    return reactive({
        connected: false,
        status: '连接中…',
        notification: '',
        players,
        myPlayerId: 0,
        currentActor: -1,
        isMyTurn: false,
        validActions: [],
        gameStarted: false,
        tilesLeft: null,
        gameEnded: false,
        debug: false
    });
}

// ---------------------------------------------------------------------------
// 事件处理（按类型分发）
// ---------------------------------------------------------------------------
function createEventHandlers(state) {
    const resetPlayers = () => {
        state.players.forEach(p => {
            p.tehai = [];
            p.discards = [];
            p.melds = [];
            p.score = INIT_SCORE;
            p.dingQueSuit = null;
        });
    };

    return {
        start_game() {
            state.gameStarted = true;
            state.gameEnded = false;
            state.status = '对局开始';
            state.tilesLeft = null;
            resetPlayers();
        },

        start_kyoku(event) {
            state.gameStarted = true;
            state.status = '对局开始';
            const { scores, tehais } = event;
            if (scores?.length === 4) scores.forEach((s, i) => (state.players[i].score = s));
            state.players.forEach(p => (p.dingQueSuit = null));
            if (tehais?.length === 4) {
                tehais.forEach((hand, i) => {
                    state.players[i].tehai = [...hand];
                    if (i === state.myPlayerId) sortHand(state.players[i].tehai);
                });
            }
        },

        tsumo(event) {
            const { actor, pai } = event;
            state.status = `玩家${actor} 摸牌`;
            state.currentActor = actor;
            state.players[actor].tehai.push(pai);
            if (actor === state.myPlayerId) {
                state.isMyTurn = true;
                state.status = '轮到你出牌';
                state.validActions = [
                    { label: '自摸', payload: { type: 'hora', actor: state.myPlayerId, target: state.myPlayerId }, class: 'btn-action btn-win' }
                ];
            }
        },

        dahai(event) {
            const { actor, pai } = event;
            state.status = `玩家${actor} 出牌`;
            state.currentActor = actor;
            const hand = state.players[actor].tehai;
            const idx = hand.indexOf(pai);
            if (idx !== -1) hand.splice(idx, 1);
            else hand.pop();
            state.players[actor].discards.push(pai);
            if (actor === state.myPlayerId) sortHand(hand);

            const me = state.myPlayerId;
            if (actor !== me && state.players[me].dingQueSuit != null) {
                const myHand = state.players[me].tehai;
                const pon = getPonConsumed(myHand, pai);
                const minkan = getMinkanConsumed(myHand, pai);
                const actions = [
                    { label: '荣和', payload: { type: 'hora', actor: me, target: actor }, class: 'btn-action btn-win' },
                    { label: '过', payload: { type: 'none' }, class: 'btn-action' }
                ];
                if (pon.length >= 2) actions.splice(1, 0, { label: '碰', payload: { type: 'pon', actor: me, target: actor, pai, consumed: pon }, class: 'btn-action' });
                if (minkan.length >= 3) actions.splice(1, 0, { label: '杠', payload: { type: 'daiminkan', actor: me, target: actor, pai, consumed: minkan }, class: 'btn-action' });
                state.validActions = actions;
            }
        },

        pon(event) { handleMeld(event); },
        daiminkan(event) { handleMeld(event); },
        chi(event) { handleMeld(event); },
        ankan(event) { handleMeld(event); },
        kakan(event) { handleMeld(event); }
    };

    function handleMeld(event) {
        const { type, actor, target, pai, consumed } = event;
        const list = Array.isArray(consumed) ? consumed : [];
        list.forEach(c => {
            const hand = state.players[actor].tehai;
            const i = hand.indexOf(c);
            if (i !== -1) hand.splice(i, 1);
            else hand.pop();
        });
        state.players[actor].melds.push({ type, pai: pai ?? list[0], consumed: list });
        if (target != null && state.players[target]?.discards?.length) state.players[target].discards.pop();
        state.currentActor = actor;
        if (actor === state.myPlayerId) {
            state.isMyTurn = true;
            state.validActions = [];
        }
    }
}

function applyEvent(state, event, handlers) {
    const { type } = event;
    if (type === 'ding_que') {
        state.status = `玩家${event.actor} 定缺`;
        state.players[event.actor].dingQueSuit = event.suit;
        return;
    }
    if (type === 'hora') {
        state.notification = `和牌！玩家${event.actor}`;
        if (event.deltas?.length === 4) event.deltas.forEach((d, i) => (state.players[i].score += d));
        setTimeout(() => (state.notification = ''), 5000);
        return;
    }
    if (type === 'game_over' || type === 'end_game') {
        state.gameEnded = true;
        state.notification = '对局结束';
        return;
    }
    const fn = handlers[type];
    if (fn) fn(event);
}

// ---------------------------------------------------------------------------
// 阶段判断（定缺等）
// ---------------------------------------------------------------------------
function evaluatePhase(state, events) {
    const idx = events.findLastIndex(e => e.type === 'start_kyoku');
    if (idx === -1) return;
    const after = events.slice(idx + 1);
    const hasPlay = after.some(e => ['dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(e.type));
    if (hasPlay) return;

    const me = state.myPlayerId;
    if (!state.players[me].dingQueSuit) {
        state.status = '请选择定缺花色';
        state.validActions = [
            { label: '万', class: 'btn-action btn-man', payload: { type: 'ding_que', actor: me, suit: 'man' } },
            { label: '筒', class: 'btn-action btn-pin', payload: { type: 'ding_que', actor: me, suit: 'pin' } },
            { label: '条', class: 'btn-action btn-sou', payload: { type: 'ding_que', actor: me, suit: 'sou' } }
        ];
    } else {
        state.status = state.isMyTurn ? '轮到你出牌' : '等待其他玩家定缺…';
        state.validActions = [];
    }
}

// ---------------------------------------------------------------------------
// Vue App（字符串模板，不依赖 in-DOM，避免解析/属性未定义问题）
// ---------------------------------------------------------------------------
const APP_TEMPLATE = `
<div class="app-root" :class="{ 'app-root--over': state.gameEnded }">
  <header class="game-header bar">
    <span class="bar-title">血战到底</span>
    <span class="bar-status">{{ state.status }}</span>
    <button v-if="!state.gameStarted || state.gameEnded" type="button" class="bar-btn" @click="tryStartGame">
      {{ state.gameEnded ? '再来一局' : '开始对局' }}
    </button>
  </header>
  <main class="main-content">
    <div class="game-board-container">
      <div class="game-board">
        <div class="center-info">
          <span v-if="state.tilesLeft != null" class="center-logo">剩 {{ state.tilesLeft }} 张</span>
          <span v-else class="center-logo">🀄</span>
        </div>
        <template v-for="z in playerZones" :key="z.zone">
          <div class="player-area" :class="['player-' + z.seat, { 'is-turn': state.currentActor === z.seat }]">
            <div class="kawa-area">
              <div v-for="(tile, ti) in state.players[z.seat].discards" :key="'d-' + ti" class="tile-wrapper">
                <div class="tile"><img :src="getPaiImage(tile, getDisplayPose(z))" class="tile-img" alt=""></div>
              </div>
            </div>
            <div class="player-info-and-hand">
              <div class="player-info-card">
                <div class="name">{{ state.players[z.seat].name }}</div>
                <div class="score" :class="{ positive: state.players[z.seat].score >= 60000, negative: state.players[z.seat].score < 60000 }">{{ state.players[z.seat].score }}</div>
                <div class="player-status-row">
                  <span v-if="state.players[z.seat].dingQueSuit" class="badge dingque" :class="state.players[z.seat].dingQueSuit">{{ suitName(state.players[z.seat].dingQueSuit) }}</span>
                </div>
              </div>
              <div class="hand-row">
                <div class="tehai-area">
                  <template v-if="z.seat !== state.myPlayerId">
                    <div v-for="n in 13" :key="'b-' + n" class="tile"><img :src="getPaiImage('?', getDisplayPose(z))" class="tile-img" alt=""></div>
                  </template>
                  <template v-else>
                    <div v-for="(tile, i) in state.players[z.seat].tehai" :key="'h-' + i" class="tile tile--mine" role="button" @click="handleTileClick(tile, i)">
                      <img :src="getPaiImage(tile, getDisplayPose(z))" class="tile-img" alt="">
                    </div>
                  </template>
                </div>
                <div class="fuuro-area" v-if="z.seat === state.myPlayerId && state.players[z.seat].melds.length">
                  <div v-for="(m, mi) in state.players[z.seat].melds" :key="'g-' + mi" class="meld-group">
                    <div v-for="t in m.consumed" :key="t" class="tile small"><img :src="getPaiImage(t, getDisplayPose(z))" class="tile-img" alt=""></div>
                    <div v-if="m.type !== 'ankan'" class="tile small"><img :src="getPaiImage(m.pai, getDisplayPose(z))" class="tile-img" alt=""></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </main>
  <div v-if="state.validActions.length > 0" class="actions" role="group">
    <button v-for="action in state.validActions" :key="action.label" type="button" class="actions-btn" :class="action.class" @click="sendAction(action)">{{ action.label }}</button>
  </div>
  <div v-if="state.notification && !state.gameEnded" class="toast" role="alert">{{ state.notification }}</div>
  <div v-if="showResultPanel" class="modal" role="dialog">
    <div class="modal-backdrop" @click.self="tryStartGame"></div>
    <div class="modal-panel">
      <h2 class="modal-title">对局结束</h2>
      <ul class="modal-list">
        <li v-for="(p, i) in state.players" :key="i" class="modal-row"><span>{{ p.name }}</span><span>{{ p.score }} 分</span></li>
      </ul>
      <button type="button" class="bar-btn modal-btn" @click="tryStartGame">再来一局</button>
    </div>
  </div>
  <pre v-if="state.debug" class="debug">{{ state.status }} {{ state.connected }} {{ state.isMyTurn }}</pre>
</div>
`;

const app = createApp({
    template: APP_TEMPLATE,
    setup() {
        const state = createInitialState();
        const handlers = createEventHandlers(state);

        let ws = null;
        const canSend = () => ws && ws.readyState === WebSocket.OPEN;

        const handleTileClick = (tile) => {
            if (!state.isMyTurn || !canSend()) return;
            ws.send(JSON.stringify({ type: 'dahai', actor: state.myPlayerId, pai: tile, tsumogiri: false }));
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
                ws.send(JSON.stringify({ type: 'start_game' }));
                state.status = '正在开局…';
                state.gameEnded = false;
            } else {
                state.status = '未连接，请刷新页面';
            }
        };

        const suitName = (suit) => (suit ? (SUIT_NAMES[suit] || suit) : '');
        const getDisplayPose = (z) => (z.seat === 3 ? 1 : z.pose);

        onMounted(() => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws/game');
            ws.onopen = () => {
                state.connected = true;
                if (!state.gameStarted) state.status = '已连接，请点击开始对局';
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'state_update') {
                    state.gameEnded = false;
                    state.players.forEach(p => { p.tehai = []; p.discards = []; p.melds = []; p.dingQueSuit = null; });
                    state.tilesLeft = null;
                    msg.data.events.forEach(ev => applyEvent(state, ev, handlers));
                    evaluatePhase(state, msg.data.events);
                } else if (msg.type === 'game_over') {
                    state.gameStarted = true;
                    state.gameEnded = true;
                    state.notification = '对局结束';
                    if (msg.scores && msg.scores.length === 4) msg.scores.forEach((s, i) => (state.players[i].score = s));
                    state.validActions = [];
                    state.isMyTurn = false;
                }
            };
            ws.onclose = () => (state.connected = false);
        });

        return {
            state,
            playerZones: PLAYER_ZONES,
            showResultPanel: computed(() => state.gameEnded),
            isMyTurn: computed(() => state.isMyTurn),
            getPaiImage,
            getDisplayPose,
            suitName,
            handleTileClick,
            sendAction,
            tryStartGame
        };
    }
});

app.mount('#app');
