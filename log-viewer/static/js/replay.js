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
        // 直接替换整个数组以确保Vue响应式更新
        window.vueApp.players = gameState.players.map(p => ({
            name: p.name,
            score: p.score,
            dingque: p.dingque
        }));
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
    
    // 排序手牌以便更好地显示（按花色和数字）
    const sortedTehai = [...player.tehai].sort((a, b) => {
        // 先按花色排序：m(萬) < p(筒) < s(条) < 字牌
        const getSuitOrder = (tile) => {
            if (tile.match(/^[1-9]m/)) return 0;
            if (tile.match(/^[1-9]p/)) return 1;
            if (tile.match(/^[1-9]s/)) return 2;
            return 3; // 字牌
        };
        const suitOrderA = getSuitOrder(a);
        const suitOrderB = getSuitOrder(b);
        if (suitOrderA !== suitOrderB) return suitOrderA - suitOrderB;
        
        // 同花色按数字排序
        const numA = parseInt(a.match(/^([1-9])/)?.[1] || '0');
        const numB = parseInt(b.match(/^([1-9])/)?.[1] || '0');
        if (numA && numB) return numA - numB;
        
        // 字牌按字母顺序
        return a.localeCompare(b);
    });
    
    // 记录最后一张牌（新摸的牌）
    const lastTile = player.tehai.length > 0 ? player.tehai[player.tehai.length - 1] : null;
    const lastTileCount = lastTile ? player.tehai.filter(t => t === lastTile).length : 0;
    let lastTileRendered = 0;
    
    sortedTehai.forEach((tile) => {
        const tileEl = document.createElement('div');
        tileEl.className = getTileClass(tile) + ' tile';
        // 标记最后一张牌（新摸的牌）为tsumo
        // 如果最后一张牌有重复，只标记最后一个出现的
        if (tile === lastTile) {
            lastTileRendered++;
            if (lastTileRendered === lastTileCount) {
                tileEl.classList.add('tile-tsumo');
            }
        }
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
        
        if (Array.isArray(meld)) {
            meld.forEach(tile => {
                const tileEl = document.createElement('div');
                tileEl.className = getTileClass(tile) + ' tile';
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
    if (!event || !event.type) {
        console.warn('Invalid event:', event);
        return;
    }
    
    const type = event.type;
    
    switch (type) {
        case 'start_game':
            gameState.players.forEach((p, i) => {
                p.name = event.names[i] || `玩家 ${i}`;
            });
            break;
            
        case 'start_kyoku':
            gameState.currentKyoku = event.kyoku || 0;
            // 初始剩余牌数：136 - 4*13(手牌) - 14(王牌) = 70，但实际游戏可能不同
            // 这里使用56作为默认值，实际应该根据游戏规则计算
            gameState.tilesLeft = 56;
            gameState.players.forEach((p, i) => {
                p.score = event.scores[i] || 25000;
                // 创建数组副本，避免引用问题
                p.tehai = (event.tehais && event.tehais[i]) ? [...event.tehais[i]] : [];
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
            if (!tsumoPlayer || !event.pai) break;
            tsumoPlayer.tehai.push(event.pai);
            // 减少剩余牌数（但要注意杠牌后从王牌补牌的情况）
            if (gameState.tilesLeft > 0) {
                gameState.tilesLeft--;
            }
            break;
            
        case 'dahai':
            const dahaiPlayer = gameState.players[event.actor];
            if (!dahaiPlayer || !event.pai) break;
            
            // 查找要打出的牌（从后往前找，因为通常打最后摸的牌）
            let tileIndex = -1;
            for (let i = dahaiPlayer.tehai.length - 1; i >= 0; i--) {
                if (dahaiPlayer.tehai[i] === event.pai) {
                    tileIndex = i;
                    break;
                }
            }
            
            if (tileIndex >= 0) {
                dahaiPlayer.tehai.splice(tileIndex, 1);
                dahaiPlayer.kawa.push(event.pai);
            } else {
                console.warn(`Player ${event.actor} tried to discard ${event.pai} but not found in hand:`, dahaiPlayer.tehai);
            }
            break;
            
        case 'pon':
            const ponPlayer = gameState.players[event.actor];
            if (!ponPlayer || !event.consumed || !Array.isArray(event.consumed)) break;
            
            // 移除手牌中的consumed牌（从后往前找，避免重复牌问题）
            const consumedCopy = [...event.consumed];
            consumedCopy.forEach(tile => {
                let idx = -1;
                for (let i = ponPlayer.tehai.length - 1; i >= 0; i--) {
                    if (ponPlayer.tehai[i] === tile) {
                        idx = i;
                        break;
                    }
                }
                if (idx >= 0) {
                    ponPlayer.tehai.splice(idx, 1);
                } else {
                    console.warn(`Player ${event.actor} pon: tile ${tile} not found in hand`);
                }
            });
            ponPlayer.fuuro.push([event.pai, ...event.consumed]);
            break;
            
        case 'daiminkan':
            const minkanPlayer = gameState.players[event.actor];
            if (!minkanPlayer || !event.consumed || !Array.isArray(event.consumed)) break;
            
            // 移除手牌中的consumed牌
            const minkanConsumed = [...event.consumed];
            minkanConsumed.forEach(tile => {
                let idx = -1;
                for (let i = minkanPlayer.tehai.length - 1; i >= 0; i--) {
                    if (minkanPlayer.tehai[i] === tile) {
                        idx = i;
                        break;
                    }
                }
                if (idx >= 0) {
                    minkanPlayer.tehai.splice(idx, 1);
                } else {
                    console.warn(`Player ${event.actor} daiminkan: tile ${tile} not found in hand`);
                }
            });
            minkanPlayer.fuuro.push([event.pai, ...event.consumed]);
            // 杠牌会减少剩余牌数（从王牌中补一张）
            // 但这里不减少tilesLeft，因为已经通过tsumo减少了
            break;
            
        case 'ankan':
            const ankanPlayer = gameState.players[event.actor];
            if (!ankanPlayer || !event.consumed || !Array.isArray(event.consumed)) break;
            
            // 移除手牌中的consumed牌（暗杠是4张相同的牌）
            const ankanConsumed = [...event.consumed];
            ankanConsumed.forEach(tile => {
                let idx = -1;
                for (let i = ankanPlayer.tehai.length - 1; i >= 0; i--) {
                    if (ankanPlayer.tehai[i] === tile) {
                        idx = i;
                        break;
                    }
                }
                if (idx >= 0) {
                    ankanPlayer.tehai.splice(idx, 1);
                } else {
                    console.warn(`Player ${event.actor} ankan: tile ${tile} not found in hand`);
                }
            });
            ankanPlayer.fuuro.push([...event.consumed]); // 创建副本
            break;
            
        case 'kakan':
            const kakanPlayer = gameState.players[event.actor];
            if (!kakanPlayer || !event.pai) break;
            
            // 从手牌中移除要加杠的牌
            let kakanIdx = -1;
            for (let i = kakanPlayer.tehai.length - 1; i >= 0; i--) {
                if (kakanPlayer.tehai[i] === event.pai) {
                    kakanIdx = i;
                    break;
                }
            }
            
            if (kakanIdx >= 0) {
                kakanPlayer.tehai.splice(kakanIdx, 1);
                // 查找对应的pon（3张牌的meld，且包含event.pai）
                let found = false;
                for (let meld of kakanPlayer.fuuro) {
                    if (meld.length === 3 && meld.includes(event.pai)) {
                        // 找到对应的pon，添加第4张牌变成kan
                        meld.push(event.pai);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    console.warn(`Player ${event.actor} kakan: pon with ${event.pai} not found`);
                }
            } else {
                console.warn(`Player ${event.actor} kakan: tile ${event.pai} not found in hand`);
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
            
        case 'end_kyoku':
            // 局结束，不需要特殊处理，状态保持
            break;
            
        case 'end_game':
            // 游戏结束
            break;
            
        case 'none':
            // 无操作事件，跳过
            break;
            
        default:
            // 未知事件类型，记录但不处理
            console.warn('Unknown event type:', type, event);
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
    
    // Process events up to current index (if any)
    for (let i = 0; i <= gameState.currentEventIndex && i < gameState.events.length; i++) {
        processEvent(gameState.events[i]);
    }
    
    updateEventDisplay();
    updateDisplay(); // Update game board display
    
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
    updateDisplay(); // Update game board display
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

// Log list management
let logListUpdateInterval = null;

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function loadLogList() {
    try {
        const response = await fetch('/api/logs');
        const data = await response.json();
        
        if (data.error) {
            console.error('Failed to load log list:', data.error);
            return;
        }
        
        // Update info
        const infoEl = document.getElementById('log-list-info');
        if (data.cached) {
            const updateTime = data.last_update ? new Date(data.last_update).toLocaleString() : '未知';
            const cachedInfo = data.total_cached ? ` (内存缓存: ${data.total_cached})` : '';
            infoEl.textContent = `目录: ${data.directory || '未知'} | 最后更新: ${updateTime} | 显示 ${data.logs.length} 个文件${cachedInfo}`;
        } else {
            infoEl.textContent = `共 ${data.logs.length} 个文件`;
        }
        
        // Render log list
        const logListEl = document.getElementById('log-list');
        logListEl.innerHTML = '';
        
        if (data.logs.length === 0) {
            logListEl.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">没有找到日志文件</div>';
            return;
        }
        
        data.logs.forEach(log => {
            const item = document.createElement('div');
            item.className = 'log-item';
            item.innerHTML = `
                <div class="log-item-name">${log.name}</div>
                <div class="log-item-meta">
                    <span>${log.mtime_str || '未知时间'}</span>
                    <span class="log-item-size">${formatFileSize(log.size)}</span>
                </div>
            `;
            
            item.addEventListener('click', () => {
                // For cached logs, always use the full path (log.path) which is the cache key
                // This ensures the backend can find it in the cache
                const path = log.path || log.cache_key || log.relative_path || log.name;
                document.getElementById('log-path-input').value = path;
                
                // Always use the full path for cached logs
                if (log.cached) {
                    console.log('Loading from memory cache:', path);
                }
                loadLogFile(path);
            });
            
            logListEl.appendChild(item);
        });
        
        // Show log list container
        document.getElementById('log-list-container').style.display = 'block';
    } catch (error) {
        console.error('Error loading log list:', error);
    }
}

function startLogListAutoRefresh() {
    // Load immediately
    loadLogList();
    
    // Then refresh every 10 seconds
    if (logListUpdateInterval) {
        clearInterval(logListUpdateInterval);
    }
    logListUpdateInterval = setInterval(loadLogList, 10000);
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
    
    // Refresh logs button
    document.getElementById('refresh-logs-btn').addEventListener('click', () => {
        loadLogList();
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
    
    // Start auto-refresh for log list
    startLogListAutoRefresh();
});

async function loadLogFile(path) {
    try {
        // Encode the path properly for URL
        // Handle both absolute and relative paths
        let encodedPath = path;
        if (path.includes('/')) {
            // For paths with slashes, encode each segment separately
            encodedPath = path.split('/').map(segment => encodeURIComponent(segment)).join('/');
        } else {
            encodedPath = encodeURIComponent(path);
        }
        
        const response = await fetch(`/api/log/${encodedPath}`);
        const logData = await response.json();
        
        if (logData.error) {
            // If file not found, suggest refreshing the log list
            let errorMsg = logData.error;
            if (errorMsg.includes('not found') || errorMsg.includes('deleted')) {
                errorMsg += '\n\n提示: 文件可能已被删除。请点击"刷新日志列表"按钮，然后从缓存中加载。';
            }
            alert('加载失败: ' + errorMsg);
            return;
        }
        
        loadLog(logData);
    } catch (error) {
        alert('加载失败: ' + error.message);
    }
}
