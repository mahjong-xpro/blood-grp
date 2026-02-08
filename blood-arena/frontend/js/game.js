// Game logic for Blood Arena

window.addEventListener("load", function () {
    console.log("Initializing Game Logic (window.load)...");

    // Connect to WebSocket
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/game`);
    const statusDiv = document.getElementById("status");

    // Add global reference for debugging
    window.ws = ws;

    ws.onopen = () => {
        console.log("Connected to server");
        if (statusDiv) statusDiv.textContent = "Connected. Waiting for game...";
        // Auto start game for MVP
        ws.send(JSON.stringify({ type: "start_game" }));
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (msg.type === "state_update") {
            handleStateUpdate(msg.data);
        } else if (msg.type === "game_over") {
            alert("Game Over! Scores: " + JSON.stringify(msg.scores));
        }
    };

    let myPlayerId = 0; // Default to 0 for MVP

    function handleStateUpdate(stateData) {
        const events = stateData.events;

        // Clear existing logs in archive_player logic
        // We access the global 'kyokus' variable exposed by archive_player.js
        // Note: We need to make sure we are modifying the array that archive_player uses.
        // 'window.kyokus = kyokus' in archive_player.js means they share the reference?
        // If I do 'kyokus = []', I am replacing the reference in game.js scope (if declared).
        // archive_player.js has 'var kyokus'.
        // If I assign to window.kyokus, does it update internal one?
        // In archive_player.js: 'kyokus = []' at top level.
        // If I say 'window.kyokus.length = 0', it clears it in place. Safer.
        if (window.kyokus) {
            window.kyokus.length = 0;
        } else {
            console.error("kyokus not found on window");
            return;
        }

        // Replay all events to reconstruct state using archive_player's logic
        for (const action of events) {
            try {
                // archive_player's loadAction handles creating kyoku objects on 'start_kyoku'
                loadAction(action);
            } catch (e) {
                console.error("Error loading action:", action, e);
            }
        }

        // Update pointers
        if (window.kyokus.length > 0) {
            currentKyokuId = window.kyokus.length - 1;
            const currentKyoku = window.kyokus[currentKyokuId];
            currentActionId = currentKyoku.actions.length - 1;

            // Render the final state
            renderCurrentAction();

            // Check turn
            checkMyTurn(events);
        }
    }

    function checkMyTurn(events) {
        if (events.length === 0) return;
        const lastEvent = events[events.length - 1];

        const controlPanel = document.getElementById("controls");
        controlPanel.innerHTML = ""; // Clear old buttons

        // Ding Que Phase Detection
        // Condition: StartKyoku exists, NO Tsumo exists, and *I* haven't done DingQue yet.
        const hasStartKyoku = events.some(e => e.type === 'start_kyoku');
        const hasTsumo = events.some(e => e.type === 'tsumo');

        if (hasStartKyoku && !hasTsumo) {
            const hasMyDingQue = events.some(e => e.type === 'ding_que' && e.actor === myPlayerId);
            if (!hasMyDingQue) {
                statusDiv.textContent = "Please select a suit to void (Ding Que).";
                showDingQueButtons();
                return;
            } else {
                statusDiv.textContent = "Waiting for other players to Ding Que...";
                return;
            }
        }

        // Normal Turn Logic
        if (lastEvent.type === "tsumo" && lastEvent.actor === myPlayerId) {
            // TURN: Self Tsumo
            statusDiv.textContent = "Your Turn! Click a tile to discard.";
            enableDiscardInteraction();

            // Add buttons for Tsumo / Kan / Reach
            // Simple logic: if can win (we don't know without libblood logic), show button?
            // For MVP, human must valid move. 
            // We can add a "Tsumo" button anyway, backend will reject if invalid.
            addActionButton("Tsumo (Win)", { type: "hora", actor: myPlayerId, target: myPlayerId });
            addActionButton("Reach", { type: "reach", actor: myPlayerId });
            addActionButton("Ankan (Select)", { type: "ankan", actor: myPlayerId, consumed: [] }); // Placeholder

            // Kan (Ankan/Kakan) - tricky to differentiate without selection
            // addActionButton("Kan", { type: "kan" ... });
        }
        else if (lastEvent.type === "dahai" && lastEvent.actor !== myPlayerId) {
            // TURN: Others Discard
            statusDiv.textContent = "Opponent Discarded.";
            // Show Pon/Kan/Ron buttons
            addActionButton("Ron", { type: "hora", actor: myPlayerId, target: lastEvent.actor });
            addActionButton("Pon", { type: "pon", actor: myPlayerId, target: lastEvent.actor, pai: lastEvent.pai, consumed: [] });
            addActionButton("Kan", { type: "daiminkan", actor: myPlayerId, target: lastEvent.actor, pai: lastEvent.pai });

            // Add Pass button
            addActionButton("Pass", { type: "none" });
        }
        else {
            statusDiv.textContent = "Waiting for opponents...";
        }
    }

    function showDingQueButtons() {
        // Enums from libblood: Man, Pin, Sou -> "man", "pin", "sou"
        const suits = ["man", "pin", "sou"];
        const labels = ["Man (Wan)", "Pin (Tong)", "Sou (Tiao)"];
        const colors = ["#d32f2f", "#1976d2", "#388e3c"]; // Red, Blue, Green hints

        suits.forEach((suit, idx) => {
            const btn = document.createElement("button");
            btn.textContent = labels[idx];
            btn.style.backgroundColor = colors[idx];
            btn.style.color = "white";
            btn.style.margin = "5px";
            btn.onclick = () => {
                sendAction({
                    type: "ding_que",
                    actor: myPlayerId,
                    suit: suit
                });
            };
            document.getElementById("controls").appendChild(btn);
        });
    }

    function enableDiscardInteraction() {
        // Attach click handlers to my tiles
        // archive_player renders tiles as <img> with specific IDs or classes
        // We need to select them.
        // players.tehais is the ID prefix?
        // See renderPais in archive_player.js

        // In archive_player.js, tehais are images in `div.tehai-container`
        // We can iterate DOM

        // Hacky selection for MVP
        // Assuming Viewpoint 0 is always at bottom
        // selector: .tehai-container img

        // But we need to know WHICH tile index corresponds to which image.
        // archive_player renders them in order.

        // IMPORTANT: archive_player.js might re-render on `renderCurrentAction`, removing listeners.
        // So we must attach listeners AFTER render.

        // We probably need to refine this to only select MY tiles (viewpoint)
        // Dytem.players.at(0).tehais... but direct DOM manipulation is easier for MVP event binding

        // Wait, renderPais (archive_player.js) replaces contents.
        // We can use event delegation on the container!
        // The container for my tiles (player 0 view)

        // Dytem creates divs with class 'player-0', 'player-1', etc.
        // And inside it 'tehai-container'.

        // Viewpoint 0 is always player-0 div (bottom)? 
        // Dytem.players.at(0) corresponds to (i - currentViewpoint + 4) % 4
        // If currentViewpoint is 0, then player 0 is index 0.

        // So we want the '.player-0 .tehai-container'

        const myTehaiContainer = document.querySelector(".player-0 .tehai-container");
        if (myTehaiContainer) {
            // Remove old listeners? Event delegation avoids this.
            // But we can just overwrite onclick of images or use container

            // Let's iterate images
            const tehaiImages = myTehaiContainer.querySelectorAll("img.pai");
            tehaiImages.forEach((img, index) => {
                img.onclick = () => {
                    // We need to know the tile value.
                    // The src is "files/images/p_ms1.gif" etc.
                    // We can't easily parse it back without map.

                    // BETTER: Look at board state.
                    // board.players[myPlayerId].tehai[index]
                    // We need access to 'board'.
                    // loadAction updates 'board' internally in archive_player.js.
                    // It is NOT exposed globally unless we hack it.

                    // BUT, we reconstructed 'kyokus'.
                    // const currentKyoku = window.kyokus[window.currentKyokuId];
                    // The last action has 'board' attached!
                    // const lastAction = currentKyoku.actions[currentKyoku.actions.length-1];
                    // const myTehai = lastAction.board.players[myPlayerId].tehais;

                    // Let's get it
                    const currentKyoku = window.kyokus[window.currentKyokuId];
                    const lastAction = currentKyoku.actions[currentKyoku.actions.length - 1];
                    if (lastAction && lastAction.board) {
                        const myTehai = lastAction.board.players[myPlayerId].tehais;
                        const paiVal = myTehai[index];

                        if (window.onTileClick) {
                            window.onTileClick(paiVal, index);
                        }
                    }
                };
                img.style.cursor = "pointer";
            });
        }
    }

    // Override renderCurrentAction/renderPai to attach listeners?
    // Or just attach after render call in handleStateUpdate.

    function addActionButton(label, actionObj) {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.onclick = () => {
            sendAction(actionObj);
        };
        document.getElementById("controls").appendChild(btn);
    }

    function sendAction(actionObj) {
        console.log("Sending action:", actionObj);
        ws.send(JSON.stringify(actionObj));
        // Clear controls to prevent double submit
        document.getElementById("controls").innerHTML = "";
        statusDiv.textContent = "Action sent. Waiting...";
    }

    // Tile click handler (to be attached to DOM)
    window.onTileClick = (paiVal, index) => {
        // Only valid if it's my turn (TSUMO)
        // Send Dahai action
        sendAction({
            type: "dahai",
            actor: myPlayerId,
            pai: paiVal,
            tsumogiri: false // TODO: check if tsumogiri
        });
    };
});
