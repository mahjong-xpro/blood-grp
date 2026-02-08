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
            console.warn("Unknown pai format:", pai);
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
            status: "Connecting...",
            notification: "", // Big center text

            // Game Data
            players: [
                { name: "Me", score: 25000, tehai: [], discards: [], melds: [], dingQueSuit: null }, // 0
                { name: "CPU 1", score: 25000, tehai: [], discards: [], melds: [], dingQueSuit: null }, // 1
                { name: "CPU 2", score: 25000, tehai: [], discards: [], melds: [], dingQueSuit: null }, // 2
                { name: "CPU 3", score: 25000, tehai: [], discards: [], melds: [], dingQueSuit: null }  // 3
            ],
            doraMarkers: [],

            // Turn Logic
            myPlayerId: 0,
            currentActor: -1,
            isMyTurn: false,
            validActions: []
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

        // --- Event Processor ---
        const handleEvent = (event) => {
            console.log("Event:", event);
            const { type, actor, target, pai, consumed, scores, tehais, suit } = event;

            // General State Update
            if (type === 'start_game') {
                state.status = "Game Started";
                state.players.forEach(p => {
                    p.tehai = []; p.discards = []; p.melds = []; p.score = 25000; p.dingQueSuit = null;
                });
                state.doraMarkers = [];
            }
            else if (type === 'start_kyoku') {
                state.status = `Kyoku ${event.kyoku} Started`;
                state.doraMarkers = [event.dora_marker];

                // Set Scores
                if (scores) scores.forEach((s, i) => state.players[i].score = s);

                // Set Hands
                // tehai is array of 4 arrays. 
                // Index 0 is player 0 (Me).
                if (tehais) {
                    tehais.forEach((hand, i) => {
                        state.players[i].tehai = [...hand];
                        // Sort my hand
                        if (i === state.myPlayerId) sortHand(state.players[i].tehai);
                    });
                }
            }
            else if (type === 'tsumo') {
                state.status = `Player ${actor} Tsumo`;
                state.currentActor = actor;

                // Add to hand
                state.players[actor].tehai.push(pai);

                // If me, enable turn
                if (actor === state.myPlayerId) {
                    state.isMyTurn = true;
                    state.status = "YOUR TURN";
                    state.validActions = [
                        { label: "Tsumo", type: "hora", payload: { type: "hora", actor: state.myPlayerId, target: state.myPlayerId }, class: "btn-action btn-win" },
                        // Check for Reach/Kan? Backend handles legality, we just offer buttons if heuristic says maybe?
                        // For MVP, simplistic: Always show Reach if closed hand? Too complex.
                        // Let's just rely on click-to-discard for now.
                    ];
                    // Add Reach/Kan buttons manually for MVP test
                }
            }
            else if (type === 'dahai') {
                state.status = `Player ${actor} Discard`;
                state.currentActor = actor;

                // Remove from hand (Value match for me, just pop for others if '?' )
                const pIdx = state.players[actor].tehai.indexOf(pai);
                if (pIdx !== -1) state.players[actor].tehai.splice(pIdx, 1);
                else if (actor !== state.myPlayerId) state.players[actor].tehai.pop(); // Remove unknown

                // Add to discards
                state.players[actor].discards.push(pai);

                // Re-sort hand (optional, keeps it neat)
                if (actor === state.myPlayerId) sortHand(state.players[actor].tehai);

                // Check for Call opportunities (if not me)
                if (actor !== state.myPlayerId && state.players[state.myPlayerId].dingQueSuit !== null) {
                    state.validActions = [
                        { label: "Ron", type: "hora", payload: { type: "hora", actor: state.myPlayerId, target: actor }, class: "btn-action btn-win" },
                        { label: "Pon", type: "pon", payload: { type: "pon", actor: state.myPlayerId, target: actor, pai: pai, consumed: [] }, class: "btn-action" },
                        { label: "Kan", type: "daiminkan", payload: { type: "daiminkan", actor: state.myPlayerId, target: actor, pai: pai }, class: "btn-action" },
                        { label: "Pass", type: "none", payload: { type: "none" }, class: "btn-action" }
                    ];
                }
            }
            else if (type === 'pon' || type === 'daiminkan' || type === 'chi') {
                // Remove consumed from actor's hand
                consumed.forEach(c => {
                    const idx = state.players[actor].tehai.indexOf(c);
                    if (idx !== -1) state.players[actor].tehai.splice(idx, 1);
                    else state.players[actor].tehai.pop();
                });
                // Add to melds
                state.players[actor].melds.push({ type, pai, consumed });

                // Remove pai from target's discard (it was taken!)
                // The last discard of target
                const targetDiscards = state.players[target].discards;
                if (targetDiscards.length > 0) targetDiscards.pop();

                state.currentActor = actor;
                if (actor === state.myPlayerId) {
                    state.isMyTurn = true; // After pon, it's my turn to discard
                    state.validActions = [];
                }
            }
            else if (type === 'ding_que') {
                state.status = `Player ${actor} Ding Que: ${suit}`;
                state.players[actor].dingQueSuit = suit;
            }
            else if (type === 'hora') {
                state.notification = `RON / TSUMO! Player ${actor}`;
                if (event.deltas && event.deltas.length === 4) {
                    event.deltas.forEach((d, i) => state.players[i].score += d);
                }
                setTimeout(() => state.notification = "", 3000);
            }
            else if (type === 'game_over') {
                state.notification = "GAME SET";
            }

            // Ding Que Prompt Logic
            // If StartKyoku happened, no Tsumo yet, and I haven't picked DingQue
            // We need to track "Has StartKyoku happened in this cycle?" 
            // Simplistic check: If my hand has 13/14 tiles and no DingQueSuit set?
            // "ding_que" event resets per kyoku? No, state.players[i].dingQueSuit needs reset on start_kyoku.
            // (Handled in start_game, but start_kyoku should also reset it? Yes)

            const noTsumoYet = state.players.every(p => p.discards.length === 0 && p.melds.length === 0); // Approx
            // StartKyoku sets tehai. 
            if (type === 'start_kyoku') {
                state.players.forEach(p => p.dingQueSuit = null);
                // Prompt immediately
                state.status = "Select Void Suit";
                state.validActions = [
                    { label: "Man", class: "btn-action btn-man", payload: { type: "ding_que", actor: state.myPlayerId, suit: "man" } },
                    { label: "Pin", class: "btn-action btn-pin", payload: { type: "ding_que", actor: state.myPlayerId, suit: "pin" } },
                    { label: "Sou", class: "btn-action btn-sou", payload: { type: "ding_que", actor: state.myPlayerId, suit: "sou" } }
                ];
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

        // --- Setup ---
        onMounted(() => {
            const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);
            ws.onopen = () => {
                state.connected = true;
                ws.send(JSON.stringify({ type: "start_game" }));
            };
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === "state_update") {
                    // Replay all events? Or just delta?
                    // backend sends FULL log usually.
                    // We must clear state and replay for robustness.
                    // MVP Optimization: Replay all.

                    // Reset Logic
                    // We can't easily reset partial state if we process incrementally.
                    // Let's implement full replay for every update to ensure sync.

                    // Reset State
                    // state.players.forEach...
                    // But 'start_game' event is usually first.
                    // Let's just process the NEW events?
                    // Does backend send ALL events every time?
                    // HumanEngine._reconstruct_state returns { events: events_list }
                    // Yes, it sends ACCUMULATED list.

                    // So we must CLEAR and REPLAY.
                    state.players.forEach(p => {
                        p.tehai = []; p.discards = []; p.melds = []; p.dingQueSuit = null;
                        // Score persists across kyokus, so careful resetting score.
                    });
                    state.doraMarkers = [];
                    // But scores? Scores are in start_kyoku.

                    msg.data.events.forEach(handleEvent);
                }
            };
        });

        return {
            state,
            getPaiImage,
            handleTileClick,
            sendAction
        };
    }
});

app.mount("#app");
