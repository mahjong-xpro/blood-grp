/**
 * 血战到底 - 单入口界面，结构清晰，无多组件嵌套
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
        const state = reactive({
            connected: false,
            myPlayerId: 0,
            currentActor: -1,
            scores: [25000, 25000, 25000, 25000],
            tehai: [],
            discards: [[], [], [], []],
            agari: [false, false, false, false],
            tilesLeft: 108,
            gameEnded: false
        });

        const analysis = reactive({ candidates: [], best_action: null });
        const ui = reactive({ selectedIdx: -1 });
        let ws = null;

        const connect = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);
            ws.onopen = () => { state.connected = true; };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'state_update') {
                    const d = msg.data;
                    if (d.analysis) {
                        analysis.candidates = d.analysis.candidates || [];
                        analysis.best_action = d.analysis.best_action || null;
                    }
                    if (d.events) replayEvents(d.events);
                } else if (msg.type === 'game_over') {
                    state.gameEnded = true;
                    alert(`对局结束 分数: ${msg.scores?.join(', ')}`);
                }
            };
            ws.onclose = () => {
                state.connected = false;
                setTimeout(connect, 3000);
            };
        };

        const send = (data) => {
            if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(data));
        };

        function replayEvents(events) {
            state.discards = [[], [], [], []];
            state.tehai = [];
            state.agari = [false, false, false, false];
            state.tilesLeft = 56;
            for (const ev of events) {
                switch (ev.type) {
                    case 'start_game':
                        state.gameEnded = false;
                        state.agari = [false, false, false, false];
                        break;
                    case 'start_kyoku':
                        state.tehai = ev.tehai || [];
                        if (ev.scores) state.scores = ev.scores;
                        break;
                    case 'tsumo':
                        state.currentActor = ev.actor;
                        state.tilesLeft = Math.max(0, state.tilesLeft - 1);
                        if (ev.actor === state.myPlayerId && ev.pai) state.tehai.push(ev.pai);
                        break;
                    case 'dahai':
                        state.currentActor = (ev.actor + 1) % 4;
                        if (!state.discards[ev.actor]) state.discards[ev.actor] = [];
                        state.discards[ev.actor].push(ev.pai);
                        if (ev.actor === state.myPlayerId) {
                            const i = state.tehai.lastIndexOf(ev.pai);
                            if (i > -1) state.tehai.splice(i, 1);
                        }
                        break;
                    case 'pon':
                    case 'chi':
                    case 'daiminkan':
                    case 'kakan':
                    case 'ankan':
                        state.currentActor = ev.actor;
                        if (ev.actor === state.myPlayerId && ev.consumed)
                            ev.consumed.forEach(t => {
                                const i = state.tehai.lastIndexOf(t);
                                if (i > -1) state.tehai.splice(i, 1);
                            });
                        break;
                    case 'agari':
                        state.agari[ev.actor] = true;
                        break;
                    case 'ryukyoku':
                        state.gameEnded = true;
                        break;
                }
            }
        }

        function player(offset) {
            const id = (state.myPlayerId + offset) % 4;
            return {
                id,
                score: state.scores[id],
                isTurn: state.currentActor === id,
                agari: state.agari && state.agari[id]
            };
        }

        function hand(offset) {
            if (offset === 0) return state.tehai;
            return Array(13).fill('back');
        }

        function discards(offset) {
            return state.discards[(state.myPlayerId + offset) % 4] || [];
        }

        function onTileClick(tile, idx) {
            if (player(0).isTurn !== true) return;
            if (ui.selectedIdx === idx) {
                send({ type: 'dahai', actor: state.myPlayerId, pai: tile, tsumogiri: false });
                ui.selectedIdx = -1;
            } else {
                ui.selectedIdx = idx;
            }
        }

        function isRecommended(tile) {
            return analysis.best_action && analysis.best_action.pai === tile;
        }

        function startGame() {
            fetch('/start_game', { method: 'POST' });
            send({ type: 'start_game' });
        }

        connect();

        return {
            state,
            analysis,
            ui,
            tileSrc,
            player,
            hand,
            discards,
            onTileClick,
            isRecommended,
            startGame
        };
    },
    template: `
    <div class="arena">
        <header class="arena-header">
            <span class="arena-title">血战到底</span>
            <span class="arena-status" :class="{ connected: state.connected }">
                {{ state.connected ? '已连接' : '连接中…' }}
            </span>
            <button class="btn" @click="startGame">开始对局</button>
        </header>

        <main class="arena-main">
            <div class="board">
                <!-- 对家 -->
                <div class="zone zone-top">
                    <div class="zone-inner">
                        <div class="player-bar" :class="{ active: player(2).isTurn }">
                            <span>对家</span>
                            <span class="score">{{ player(2).score }}</span>
                            <span class="agari" v-if="player(2).agari">和</span>
                        </div>
                        <div class="hand-row">
                            <div class="tile-wrap" v-for="(t, i) in hand(2)" :key="'t'+i">
                                <img :src="tileSrc(t)" :alt="t">
                            </div>
                        </div>
                        <div class="river-row">
                            <div class="tile-wrap" v-for="(d, i) in discards(2)" :key="'dt'+i">
                                <img :src="tileSrc(d)" :alt="d">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 上家 -->
                <div class="zone zone-left">
                    <div class="zone-inner">
                        <div class="player-bar" :class="{ active: player(3).isTurn }">上家 {{ player(3).score }}</div>
                        <div class="hand-row">
                            <div class="tile-wrap" v-for="(t, i) in hand(3)" :key="'l'+i">
                                <img :src="tileSrc(t)" :alt="t">
                            </div>
                        </div>
                        <div class="river-row">
                            <div class="tile-wrap" v-for="(d, i) in discards(3)" :key="'dl'+i">
                                <img :src="tileSrc(d)" :alt="d">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 下家 -->
                <div class="zone zone-right">
                    <div class="zone-inner">
                        <div class="player-bar" :class="{ active: player(1).isTurn }">下家 {{ player(1).score }}</div>
                        <div class="hand-row">
                            <div class="tile-wrap" v-for="(t, i) in hand(1)" :key="'r'+i">
                                <img :src="tileSrc(t)" :alt="t">
                            </div>
                        </div>
                        <div class="river-row">
                            <div class="tile-wrap" v-for="(d, i) in discards(1)" :key="'dr'+i">
                                <img :src="tileSrc(d)" :alt="d">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 自己 -->
                <div class="zone zone-bottom">
                    <div class="zone-inner">
                        <div class="river-row">
                            <div class="tile-wrap" v-for="(d, i) in discards(0)" :key="'db'+i">
                                <img :src="tileSrc(d)" :alt="d">
                            </div>
                        </div>
                        <div class="hand-row">
                            <div class="tile-wrap clickable"
                                 v-for="(t, i) in hand(0)"
                                 :key="'b'+i"
                                 :class="{ selected: ui.selectedIdx === i, recommended: isRecommended(t) }"
                                 @click="onTileClick(t, i)">
                                <img :src="tileSrc(t)" :alt="t">
                            </div>
                        </div>
                        <div class="player-bar" :class="{ active: player(0).isTurn }">
                            <span>我</span>
                            <span class="score">{{ player(0).score }}</span>
                            <span class="agari" v-if="player(0).agari">和</span>
                        </div>
                    </div>
                </div>

                <!-- 中央 -->
                <div class="zone zone-center">
                    <div class="turn" :class="{ my: player(0).isTurn }">
                        {{ player(0).isTurn ? '请打牌' : '等待' }}
                    </div>
                    <div class="tiles-left">剩余 {{ state.tilesLeft }} 张</div>
                </div>

                <!-- AI 推荐 -->
                <div class="ai-bar" :class="{ hidden: !analysis.best_action || analysis.best_action.type !== 'dahai' }">
                    <span class="label">推荐打</span>
                    <div class="tile-wrap" v-if="analysis.best_action && analysis.best_action.pai">
                        <img :src="tileSrc(analysis.best_action.pai)" :alt="analysis.best_action.pai">
                    </div>
                </div>
            </div>
        </main>
    </div>
    `
});

app.mount('#app');
