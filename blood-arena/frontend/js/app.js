import { createApp, reactive } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';
import GameBoard from './components/GameBoard.js';

const app = createApp({
    components: { GameBoard },
    setup() {
        const state = reactive({
            connected: false,
            gameID: null,
            myPlayerId: 0,
            currentActor: -1,
            scores: [25000, 25000, 25000, 25000],
            tehai: [], // Array of strings e.g. ["1m", "2m"]
            discards: [[], [], [], []],
            agari: [false, false, false, false],
            tilesLeft: 108,
            gameEnded: false
        });

        const analysis = reactive({
            candidates: [],
            best_action: null
        });

        let ws = null;

        const connect = () => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);

            ws.onopen = () => {
                console.log("WebSocket Connected");
                state.connected = true;
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            };

            ws.onclose = () => {
                console.log("WebSocket Disconnected");
                state.connected = false;
                setTimeout(connect, 3000); // Auto-reconnect
            };
        };

        const send = (data) => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(data));
            }
        };

        const handleMessage = (msg) => {
            if (msg.type === 'state_update') {
                const data = msg.data;

                // Update Analysis
                if (data.analysis) {
                    analysis.candidates = data.analysis.candidates || [];
                    analysis.best_action = data.analysis.best_action;
                }

                // Update Game State
                if (data.events) {
                    replayEvents(data.events);
                }
            } else if (msg.type === 'game_over') {
                state.gameEnded = true;
                alert(`Game Over! Scores: ${msg.scores}`);
            }
        };

        const replayEvents = (events) => {
            // Reset granular state for full replay
            state.discards = [[], [], [], []];
            state.tehai = [];
            state.agari = [false, false, false, false];
            state.tilesLeft = 56; // Check this default

            // Replay logic
            for (const ev of events) {
                switch (ev.type) {
                    case 'start_game':
                        // Reset everything
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
                        if (ev.actor === state.myPlayerId) {
                            state.tehai.push(ev.pai);
                        }
                        break;
                    case 'dahai':
                        state.currentActor = (ev.actor + 1) % 4;
                        if (!state.discards[ev.actor]) state.discards[ev.actor] = [];
                        state.discards[ev.actor].push(ev.pai);
                        if (ev.actor === state.myPlayerId) {
                            const idx = state.tehai.lastIndexOf(ev.pai);
                            if (idx > -1) state.tehai.splice(idx, 1);
                        }
                        break;
                    case 'pon':
                    case 'chi':
                    case 'daiminkan':
                    case 'kakan':
                    case 'ankan':
                        state.currentActor = ev.actor;
                        if (ev.actor === state.myPlayerId && ev.consumed) {
                            ev.consumed.forEach(t => {
                                const idx = state.tehai.lastIndexOf(t);
                                if (idx > -1) state.tehai.splice(idx, 1);
                            });
                        }
                        break;
                    case 'agari':
                        state.agari[ev.actor] = true;
                        break;
                    case 'ryukyoku':
                        state.gameEnded = true;
                        break;
                }
            }
        };

        const handleAction = (event) => {
            // event from GameBoard: { type: 'dahai', pai: ... }
            if (event.type === 'dahai') {
                send({
                    type: 'dahai',
                    actor: state.myPlayerId,
                    pai: event.pai,
                    tsumogiri: false // Simple assumption for now
                });
            }
        };

        const startGame = () => {
            // Request backend to start game
            fetch('/start_game?ai_model=/Users/twosson/Mahjong/blood/data/models/latest.pth', { method: 'POST' });
            send({ type: 'start_game' });
        };

        connect();

        return {
            state,
            analysis,
            handleAction,
            startGame
        };
    },
    template: `
        <div class="app-root">
            <header>
                 <h1>Blood Arena (Professional)</h1>
                 <button class="btn btn-secondary" @click="startGame">NEW GAME</button>
            </header>
            
            <main style="flex: 1; overflow: hidden; position: relative;">
                <GameBoard 
                    :state="state" 
                    :analysis="analysis"
                    @action="handleAction" />
            </main>
        </div>
    `
});

app.mount('#app');
