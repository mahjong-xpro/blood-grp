const { createApp, reactive, onMounted } = Vue;

createApp({
    setup() {
        const state = reactive({
            status: "Connecting...",
            scores: [25000, 25000, 25000, 25000],
            validActions: [], // Array of { label, payload, class }
            isMyTurn: false,
            currentActor: -1, // Who is acting now
            notification: "", // Center screen huge text
            myPlayerId: 0
        });

        let ws = null;

        // --- WebSocket Logic ---
        const connect = () => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);

            ws.onopen = () => {
                state.status = "Connected. Starting game...";
                ws.send(JSON.stringify({ type: "start_game" }));
            };

            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                if (msg.type === "state_update") {
                    handleStateUpdate(msg.data);
                } else if (msg.type === "game_over") {
                    state.status = "Game Over";
                    state.notification = "GAME OVER";
                    state.scores = msg.scores;
                }
            };

            ws.onerror = (e) => {
                state.status = "Connection Error";
                console.error(e);
            };
        };

        // --- Game Logic ---
        const handleStateUpdate = (stateData) => {
            const events = stateData.events;

            // 1. Update Legacy Board (Dytem)
            if (window.kyokus) {
                window.kyokus.length = 0; // Clear legacy buffer
            }

            try {
                // Replay events using archive_player logic
                for (const action of events) {
                    window.loadAction(action);
                }

                // Update Dytem View
                if (window.kyokus && window.kyokus.length > 0) {
                    window.currentKyokuId = window.kyokus.length - 1;
                    const currentKyoku = window.kyokus[window.currentKyokuId];
                    window.currentActionId = currentKyoku.actions.length - 1;
                    window.renderCurrentAction();

                    // Sync Scores from last action
                    // archive_player doesn't easily expose scores in a global var, 
                    // but we can parse from 'scores' or 'deltas' in events if needed.
                    // For now, let's rely on StartKyoku event for initial, and accumulate deltas? 
                    // Or just read from Dytem DOM? Dytem updates #playerInfos.score
                    // Let's defer precise score sync or just grab from the last 'hora'/'ryukyoku' event.
                    // Actually, StartKyoku has scores.
                    const startKyoku = events.find(e => e.type === 'start_kyoku');
                    if (startKyoku) {
                        state.scores = startKyoku.scores;
                        // TODO: Apply deltas from subsequent Hora/Ryukyoku? 
                        // archive_player handles this internally for display, but doesn't export.
                        // For MVP, we might just trust the 'scores' in StartKyoku + manual deltas?
                        // Or just let Dytem show scores on board, and we show them in HUD?
                        // Let's stick to initial scores for now to be safe.
                    }
                }
            } catch (e) {
                console.error("Error updating board:", e);
            }

            // 2. Update Vue State & Interactions
            updateInteractions(events);
        };

        const updateInteractions = (events) => {
            if (!events.length) return;
            const lastEvent = events[events.length - 1];
            state.currentActor = lastEvent.actor;

            state.validActions = [];
            state.notification = "";

            // --- Ding Que Detection ---
            const hasStartKyoku = events.some(e => e.type === 'start_kyoku');
            const hasTsumo = events.some(e => e.type === 'tsumo');

            if (hasStartKyoku && !hasTsumo) {
                const hasMyDingQue = events.some(e => e.type === 'ding_que' && e.actor === state.myPlayerId);
                if (!hasMyDingQue) {
                    state.status = "Select Void Suit (Ding Que)";
                    state.notification = "SELECT VOID SUIT";
                    state.validActions = [
                        { label: "Man (Wan)", class: "btn-suit-man", payload: { type: "ding_que", actor: state.myPlayerId, suit: "man" } },
                        { label: "Pin (Tong)", class: "btn-suit-pin", payload: { type: "ding_que", actor: state.myPlayerId, suit: "pin" } },
                        { label: "Sou (Tiao)", class: "btn-suit-sou", payload: { type: "ding_que", actor: state.myPlayerId, suit: "sou" } }
                    ];
                    return;
                } else {
                    state.status = "Waiting for others...";
                    return;
                }
            }

            // --- Normal Turn Logic ---
            if (lastEvent.type === "tsumo" && lastEvent.actor === state.myPlayerId) {
                state.status = "Your Turn";
                state.isMyTurn = true;
                // Actions
                state.validActions.push({
                    label: "Tsumo",
                    class: "success",
                    payload: { type: "hora", actor: state.myPlayerId, target: state.myPlayerId }
                });
                state.validActions.push({
                    label: "Reach",
                    class: "primary",
                    payload: { type: "reach", actor: state.myPlayerId }
                });
                // Discard is handled by clicking tiles (Dytem integration needed)
            }
            else if (lastEvent.type === "dahai" && lastEvent.actor !== state.myPlayerId) {
                state.status = "Opponent Discarded";
                state.validActions.push({
                    label: "Ron",
                    class: "danger",
                    payload: { type: "hora", actor: state.myPlayerId, target: lastEvent.actor }
                });
                state.validActions.push({
                    label: "Pon",
                    class: "primary",
                    payload: { type: "pon", actor: state.myPlayerId, target: lastEvent.actor, pai: lastEvent.pai, consumed: [] }
                });
                state.validActions.push({
                    label: "Kan",
                    class: "primary",
                    payload: { type: "daiminkan", actor: state.myPlayerId, target: lastEvent.actor, pai: lastEvent.pai }
                });
                state.validActions.push({
                    label: "Pass",
                    class: "",
                    payload: { type: "none" }
                });
            } else {
                state.status = "Waiting...";
                state.isMyTurn = false;
            }
        };

        const handleAction = (payload) => {
            console.log("Sending action:", payload);
            ws.send(JSON.stringify(payload));
            state.validActions = []; // Clear buttons immediately
            state.status = "Action sent";
        };

        // --- Tile Click Handling (Dytem Integration) ---
        // We need to attach listeners to the Dytem-rendered tiles
        // Since Dytem re-renders on 'renderCurrentAction', we might need to re-attach or use delegation.
        // Game.js used a global hook. Let's replicate that.

        window.onTileClick = (paiVal, index) => {
            if (state.isMyTurn) {
                handleAction({
                    type: "dahai",
                    actor: state.myPlayerId,
                    pai: paiVal,
                    tsumogiri: false
                });
            }
        };

        // We also need the global listener setup from game.js
        // But since game.js is gone, we must reimplement the event delegation here.
        const setupTileListeners = () => {
            // Use document body delegation
            document.body.addEventListener('click', (e) => {
                if (e.target.tagName === 'IMG' && e.target.classList.contains('pai')) {
                    // Check if it's MY hand tiles
                    // The container for my tiles is .player-0 .tehai-container (assuming Viewpoint 0)
                    const container = e.target.closest('.player-0 .tehai-container');
                    if (container) {
                        const allImgs = Array.from(container.querySelectorAll('img.pai'));
                        const index = allImgs.indexOf(e.target);
                        if (index !== -1) {
                            // Recover pai value from ArchivePlayer state
                            // window.kyokus[window.currentKyokuId].actions[window.currentActionId].board.players[0].tehais[index]
                            try {
                                const currentKyoku = window.kyokus[window.currentKyokuId];
                                const lastAction = currentKyoku.actions[currentKyoku.actions.length - 1];
                                if (lastAction && lastAction.board) {
                                    const myTehai = lastAction.board.players[state.myPlayerId].tehais;
                                    const paiVal = myTehai[index];
                                    if (paiVal) {
                                        window.onTileClick(paiVal, index);
                                    }
                                }
                            } catch (err) {
                                console.error("Tile click error", err);
                            }
                        }
                    }
                }
            });
        };

        onMounted(() => {
            console.log("Vue App Mounted");
            connect();
            setupTileListeners();
        });

        return {
            ...Vue.toRefs(state),
            handleAction
        };
    }
}).mount('#app');
