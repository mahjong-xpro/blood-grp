// Mahjong Log Replay JavaScript

// Tile rendering utilities
function tileToText(tile) {
    if (!tile) return '';
    
    // Handle special tiles
    const specialTiles = {
        'E': '東', 'S': '南', 'W': '西', 'N': '北',
        'P': '白', 'F': '發', 'C': '中'
    };
    
    if (specialTiles[tile]) {
        return specialTiles[tile];
    }
    
    // Handle numbered tiles (e.g., "1m", "5p", "9s")
    const match = tile.match(/^([1-9])([mps])$/);
    if (match) {
        const [, num, suit] = match;
        const suitNames = { 'm': '萬', 'p': '筒', 's': '条' };
        return num + suitNames[suit];
    }
    
    return tile;
}

function getTileClass(tile) {
    if (!tile) return 'tile';
    
    if (tile.match(/^[1-9]m/)) return 'tile man';
    if (tile.match(/^[1-9]p/)) return 'tile pin';
    if (tile.match(/^[1-9]s/)) return 'tile sou';
    if (['E', 'S', 'W', 'N', 'P', 'F', 'C'].includes(tile)) return 'tile ji';
    
    return 'tile';
}

// Game state
let gameState = {
    events: [],
    currentEventIndex: 0,
    players: [
        { name: '玩家 0', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 1', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 2', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 3', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
    ],
    currentKyoku: 0,
    tilesLeft: 56,
    isPlaying: false,
    playSpeed: 1.0,
    playInterval: null,
};

// Display update function
function updateDisplay() {
    // Update Vue app data (reactive)
    if (window.vueApp) {
        // Update players
        gameState.players.forEach((p, i) => {
            if (window.vueApp.players[i]) {
                window.vueApp.players[i].name = p.name;
                window.vueApp.players[i].score = p.score;
                window.vueApp.players[i].dingque = p.dingque;
            }
        });
        window.vueApp.currentKyoku = gameState.currentKyoku;
        window.vueApp.tilesLeft = gameState.tilesLeft;
    }
    
    // Render board
    renderBoard();
}

function renderBoard() {
    // Render hands
    for (let i = 0; i < 4; i++) {
        renderHand(i);
        renderDiscard(i);
        renderFuuro(i);
    }
}

function renderHand(playerId) {
    const handArea = document.getElementById(`hand-${playerId}`);
    if (!handArea) return;
    
    handArea.innerHTML = '';
    const player = gameState.players[playerId];
    
    if (!player || !player.tehai) return;
    
    player.tehai.forEach(tile => {
        const tileEl = document.createElement('div');
        tileEl.className = getTileClass(tile) + ' tile-tsumo';
        tileEl.textContent = tileToText(tile);
        tileEl.title = tile;
        handArea.appendChild(tileEl);
    });
}

function renderDiscard(playerId) {
    const discardArea = document.getElementById(`discard-${playerId}`);
    if (!discardArea) return;
    
    discardArea.innerHTML = '';
    const player = gameState.players[playerId];
    
    if (!player || !player.kawa) return;
    
    player.kawa.forEach(tile => {
        const tileEl = document.createElement('div');
        tileEl.className = getTileClass(tile) + ' tile discarded';
        tileEl.textContent = tileToText(tile);
        tileEl.title = tile;
        discardArea.appendChild(tileEl);
    });
}

function renderFuuro(playerId) {
    const fuuroArea = document.getElementById(`fuuro-${playerId}`);
    if (!fuuroArea) return;
    
    fuuroArea.innerHTML = '';
    const player = gameState.players[playerId];
    
    if (!player || !player.fuuro) return;
    
    player.fuuro.forEach(meld => {
        const meldEl = document.createElement('div');
        meldEl.className = 'meld';
        meldEl.style.display = 'flex';
        meldEl.style.gap = '2px';
        meldEl.style.margin = '2px';
        
        if (Array.isArray(meld)) {
            meld.forEach(tile => {
                const tileEl = document.createElement('div');
                tileEl.className = getTileClass(tile);
                tileEl.textContent = tileToText(tile);
                tileEl.title = tile;
                meldEl.appendChild(tileEl);
            });
        }
        
        fuuroArea.appendChild(meldEl);
    });
}

// Event processing
function processEvent(event) {
    const type = event.type;
    
    switch (type) {
        case 'start_game':
            gameState.players.forEach((p, i) => {
                p.name = event.names[i] || `玩家 ${i}`;
            });
            break;
            
        case 'start_kyoku':
            gameState.currentKyoku = event.kyoku;
            gameState.tilesLeft = 56;
            gameState.players.forEach((p, i) => {
                p.score = event.scores[i];
                p.tehai = event.tehais[i] || [];
                p.kawa = [];
                p.fuuro = [];
                p.dingque = null;
            });
            break;
            
        case 'ding_que':
            const dqPlayer = gameState.players[event.actor];
            const suitNames = { 'man': '萬', 'pin': '筒', 'sou': '条' };
            dqPlayer.dingque = suitNames[event.suit] || event.suit;
            break;
            
        case 'tsumo':
            const tsumoPlayer = gameState.players[event.actor];
            tsumoPlayer.tehai.push(event.pai);
            gameState.tilesLeft--;
            break;
            
        case 'dahai':
            const dahaiPlayer = gameState.players[event.actor];
            const tileIndex = dahaiPlayer.tehai.indexOf(event.pai);
            if (tileIndex >= 0) {
                dahaiPlayer.tehai.splice(tileIndex, 1);
                dahaiPlayer.kawa.push(event.pai);
            }
            break;
            
        case 'pon':
            const ponPlayer = gameState.players[event.actor];
            event.consumed.forEach(tile => {
                const idx = ponPlayer.tehai.indexOf(tile);
                if (idx >= 0) ponPlayer.tehai.splice(idx, 1);
            });
            ponPlayer.fuuro.push([event.pai, ...event.consumed]);
            break;
            
        case 'daiminkan':
            const minkanPlayer = gameState.players[event.actor];
            event.consumed.forEach(tile => {
                const idx = minkanPlayer.tehai.indexOf(tile);
                if (idx >= 0) minkanPlayer.tehai.splice(idx, 1);
            });
            minkanPlayer.fuuro.push([event.pai, ...event.consumed]);
            break;
            
        case 'ankan':
            const ankanPlayer = gameState.players[event.actor];
            event.consumed.forEach(tile => {
                const idx = ankanPlayer.tehai.indexOf(tile);
                if (idx >= 0) ankanPlayer.tehai.splice(idx, 1);
            });
            ankanPlayer.fuuro.push(event.consumed);
            break;
            
        case 'kakan':
            const kakanPlayer = gameState.players[event.actor];
            const kakanIdx = kakanPlayer.tehai.indexOf(event.pai);
            if (kakanIdx >= 0) {
                kakanPlayer.tehai.splice(kakanIdx, 1);
                // Find the pon and convert to kan
                for (let meld of kakanPlayer.fuuro) {
                    if (meld.length === 3 && meld[0] === event.pai) {
                        meld.push(event.pai);
                        break;
                    }
                }
            }
            break;
            
        case 'hora':
            // Update scores if deltas provided
            if (event.deltas) {
                event.deltas.forEach((delta, i) => {
                    gameState.players[i].score += delta;
                });
            }
            break;
            
        case 'ryukyoku':
            // Update scores if deltas provided
            if (event.deltas) {
                event.deltas.forEach((delta, i) => {
                    gameState.players[i].score += delta;
                });
            }
            break;
    }
}

function resetGameState() {
    gameState.players = [
        { name: '玩家 0', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 1', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 2', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
        { name: '玩家 3', score: 25000, dingque: null, tehai: [], kawa: [], fuuro: [] },
    ];
    gameState.currentEventIndex = 0;
    gameState.currentKyoku = 0;
    gameState.tilesLeft = 56;
}

function loadLog(logData) {
    gameState.events = logData.events || [];
    gameState.currentEventIndex = 0;
    
    // Update player names
    if (logData.names) {
        logData.names.forEach((name, i) => {
            gameState.players[i].name = name;
        });
    }
    
    // Process all events up to current index
    resetGameState();
    updateEventDisplay();
    
    document.getElementById('event-total').textContent = gameState.events.length;
    document.getElementById('game-info').style.display = 'block';
}

function updateEventDisplay() {
    const eventIndex = gameState.currentEventIndex;
    const totalEvents = gameState.events.length;
    
    document.getElementById('event-index').textContent = eventIndex;
    document.getElementById('event-total').textContent = totalEvents;
    
    if (eventIndex < totalEvents) {
        const event = gameState.events[eventIndex];
        document.getElementById('current-event').textContent = JSON.stringify(event, null, 2);
    }
    
    // Update log display
    updateLogDisplay();
}

function updateLogDisplay() {
    const logContent = document.getElementById('log-content');
    logContent.innerHTML = '';
    
    const startIdx = Math.max(0, gameState.currentEventIndex - 10);
    const endIdx = Math.min(gameState.events.length, gameState.currentEventIndex + 10);
    
    for (let i = startIdx; i < endIdx; i++) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        if (i === gameState.currentEventIndex) {
            entry.classList.add('active');
        }
        entry.textContent = `[${i}] ${JSON.stringify(gameState.events[i])}`;
        logContent.appendChild(entry);
    }
    
    logContent.scrollTop = logContent.scrollHeight;
}

function goToEvent(index) {
    if (index < 0 || index >= gameState.events.length) return;
    
    // Reset and replay up to target index
    resetGameState();
    
    for (let i = 0; i <= index; i++) {
        processEvent(gameState.events[i]);
    }
    
    gameState.currentEventIndex = index;
    updateEventDisplay();
}

function prevEvent() {
    if (gameState.currentEventIndex > 0) {
        goToEvent(gameState.currentEventIndex - 1);
    }
}

function nextEvent() {
    if (gameState.currentEventIndex < gameState.events.length - 1) {
        goToEvent(gameState.currentEventIndex + 1);
    }
}

function firstEvent() {
    goToEvent(0);
}

function lastEvent() {
    goToEvent(gameState.events.length - 1);
}

function togglePlayPause() {
    if (gameState.isPlaying) {
        // Pause
        if (gameState.playInterval) {
            clearInterval(gameState.playInterval);
            gameState.playInterval = null;
        }
        gameState.isPlaying = false;
        document.getElementById('play-pause-btn').textContent = '▶ 播放';
    } else {
        // Play
        gameState.isPlaying = true;
        document.getElementById('play-pause-btn').textContent = '⏸ 暂停';
        
        gameState.playInterval = setInterval(() => {
            if (gameState.currentEventIndex < gameState.events.length - 1) {
                nextEvent();
            } else {
                togglePlayPause(); // Auto pause at end
            }
        }, 1000 / gameState.playSpeed);
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // File upload
    document.getElementById('upload-btn').addEventListener('click', () => {
        document.getElementById('file-input').click();
    });
    
    document.getElementById('file-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            
            const result = await response.json();
            if (result.path) {
                document.getElementById('log-path-input').value = result.path;
                loadLogFile(result.path);
            }
        } catch (error) {
            alert('上传失败: ' + error.message);
        }
    });
    
    // Load button
    document.getElementById('load-btn').addEventListener('click', () => {
        const path = document.getElementById('log-path-input').value;
        if (path) {
            loadLogFile(path);
        }
    });
    
    // Playback controls
    document.getElementById('prev-btn').addEventListener('click', prevEvent);
    document.getElementById('next-btn').addEventListener('click', nextEvent);
    document.getElementById('first-btn').addEventListener('click', firstEvent);
    document.getElementById('last-btn').addEventListener('click', lastEvent);
    document.getElementById('play-pause-btn').addEventListener('click', togglePlayPause);
    
    // Speed control
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    
    speedSlider.addEventListener('input', (e) => {
        gameState.playSpeed = parseFloat(e.target.value);
        speedValue.textContent = gameState.playSpeed.toFixed(1) + 'x';
        
        // Restart interval if playing
        if (gameState.isPlaying) {
            togglePlayPause();
            togglePlayPause();
        }
    });
});

async function loadLogFile(path) {
    try {
        const response = await fetch(`/api/log/${encodeURIComponent(path)}`);
        const logData = await response.json();
        
        if (logData.error) {
            alert('加载失败: ' + logData.error);
            return;
        }
        
        loadLog(logData);
    } catch (error) {
        alert('加载失败: ' + error.message);
    }
}
