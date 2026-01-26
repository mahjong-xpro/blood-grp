import { createApp, ref, computed, onMounted, watch, nextTick } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';

const App = {
    setup() {
        // --- State ---
        const searchQuery = ref('');
        const logList = ref([]);
        const loadingLogs = ref(false);
        const currentLogPath = ref(null);

        // Game State
        const gameLoaded = ref(false);
        const currentEventIndex = ref(0);
        const events = ref([]);
        const totalEvents = computed(() => events.value.length);
        const gameInfo = ref({});

        // Playback State
        const isPlaying = ref(false);
        const playbackSpeed = ref(1.0);
        let playInterval = null;

        // Players State (Snapshot at current index)
        const players = ref([
            { name: 'Player 0', score: 25000, dingque: null, agari: false, riichi: false, tehai: [], kawa: [], fuuro: [] },
            { name: 'Player 1', score: 25000, dingque: null, agari: false, riichi: false, tehai: [], kawa: [], fuuro: [] },
            { name: 'Player 2', score: 25000, dingque: null, agari: false, riichi: false, tehai: [], kawa: [], fuuro: [] },
            { name: 'Player 3', score: 25000, dingque: null, agari: false, riichi: false, tehai: [], kawa: [], fuuro: [] },
        ]);
        const currentKyoku = ref(0);
        const tilesLeft = ref(56);
        const currentPlayer = ref(null); // Who is acting?
        const lastDiscard = ref(null); // { actor: 0, tile: '5m' }

        // UI State
        const showRightPanel = ref(true);
        const eventListRef = ref(null);

        // --- Computed ---
        const filteredLogs = computed(() => {
            if (!searchQuery.value) return logList.value;
            const q = searchQuery.value.toLowerCase();
            return logList.value.filter(l => l.name.toLowerCase().includes(q));
        });

        const currentKyokuDisplay = computed(() => {
            // Converts 0-3 to East 1-4, 4-7 to South 1-4 etc if needed. 
            // For Bloody Battle, usually just "Kyoku X".
            return `Kyoku ${currentKyoku.value + 1}`;
        });

        const visibleEvents = computed(() => {
            // Optimization: Only show a window of events around current?
            // For now show all, virtual scrolling might be needed for huge logs.
            // But let's map them for display.
            return events.value.map((e, i) => ({
                index: i,
                type: e.type,
                actor: e.actor,
                detail: getEventDetail(e)
            }));
        });

        // --- Methods: Data Loading ---
        const refreshLogs = async () => {
            loadingLogs.value = true;
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                if (data.logs) {
                    logList.value = data.logs;
                }
            } catch (e) {
                console.error("Failed to load logs", e);
            } finally {
                loadingLogs.value = false;
            }
        };

        const loadLog = async (path) => {
            // Stop playback
            if (isPlaying.value) togglePlay();

            // normalize
            const encodedPath = path.split('/').map(encodeURIComponent).join('/');

            try {
                const res = await fetch(`/api/log/${encodedPath}`);
                const data = await res.json();

                if (data.error) throw new Error(data.error);

                // Init Game
                events.value = data.events || [];
                gameInfo.value = data;
                gameLoaded.value = true;
                currentLogPath.value = path;

                // Parse Names
                if (data.names) {
                    // Stored in metadata for reset
                    gameInfo.value.playerNames = data.names;
                }

                // Reset to start
                seekTo(0);

            } catch (e) {
                alert(`Error loading log: ${e.message}`);
            }
        };

        // --- Methods: Game Logic ---
        const resetGameState = () => {
            const names = gameInfo.value.playerNames || ['Player 0', 'Player 1', 'Player 2', 'Player 3'];
            players.value = names.map(name => ({
                name,
                score: 25000,
                dingque: null,
                dingque_suit: '',
                dingque_char: '',
                agari: false,
                riichi: false,
                tehai: [], // Array of strings
                kawa: [],
                fuuro: [], // Array of arrays
                lastAction: ''
            }));
            currentKyoku.value = 0;
            tilesLeft.value = 56; // Blood battle default?
            currentPlayer.value = null;
            lastDiscard.value = null;
        };

        const processEvent = (event) => {
            const type = event.type;
            const actor = event.actor;
            const p = actor !== undefined ? players.value[actor] : null;

            switch (type) {
                case 'start_game':
                    // Usually handled by metadata, but safe to update names
                    if (event.names) {
                        event.names.forEach((n, i) => players.value[i].name = n);
                    }
                    break;
                case 'start_kyoku':
                    currentKyoku.value = event.kyoku || 0;
                    tilesLeft.value = 56; // Reset wall
                    // Scores usually update here too?
                    if (event.scores) {
                        event.scores.forEach((s, i) => players.value[i].score = s);
                    }
                    if (event.tehais) {
                        event.tehais.forEach((hand, i) => {
                            players.value[i].tehai = [...hand];
                            // Sort hand
                            sortHand(players.value[i]);
                        });
                    }
                    // Reset round state
                    players.value.forEach(pl => {
                        pl.kawa = [];
                        pl.fuuro = [];
                        pl.dingque = null;
                        pl.agari = false;
                        pl.riichi = false;
                        pl.lastAction = '';
                    });
                    lastDiscard.value = null;
                    currentPlayer.value = null;
                    break;

                case 'ding_que':
                    if (p) {
                        const suitMap = { 'man': '萬', 'pin': '筒', 'sou': '条' };
                        const colorMap = { 'man': 'man', 'pin': 'pin', 'sou': 'sou' };
                        p.dingque = event.suit;
                        p.dingque_char = suitMap[event.suit] || event.suit[0].toUpperCase();
                        p.dingque_suit = colorMap[event.suit] || '';
                    }
                    break;

                case 'tsumo':
                    if (p && event.pai) {
                        p.tehai.push(event.pai);
                        p.lastAction = 'tsumo';
                        currentPlayer.value = actor;
                        if (tilesLeft.value > 0) tilesLeft.value--;
                    }
                    break;

                case 'dahai':
                    if (p && event.pai) {
                        removeTile(p, event.pai);
                        p.kawa.push(event.pai);
                        sortHand(p); // efficient sort
                        p.lastAction = 'dahai';
                        currentPlayer.value = actor;
                        lastDiscard.value = { actor, tile: event.pai };
                    }
                    break;

                case 'ankan':
                    if (p && event.consumed) {
                        event.consumed.forEach(t => removeTile(p, t));
                        // Mark as ankan for UI rendering (e.g. [t1, t2, t3, t4, 'ankan'])
                        // Or better: object wrapper? 
                        // Current fuuro is array of strings. Let's make it robust.
                        // We will store metadata in the array itself or change structure.
                        // Easy hack: assign property to the array? JS allows it.
                        const meld = [...event.consumed];
                        meld.type = 'ankan';
                        p.fuuro.push(meld);
                        p.lastAction = 'ankan';
                    }
                    break;

                case 'daiminkan':
                    if (p && event.consumed) {
                        event.consumed.forEach(t => removeTile(p, t));
                        const meld = [event.pai, ...event.consumed];
                        meld.type = 'minkan';
                        p.fuuro.push(meld);
                        p.lastAction = 'daiminkan';
                        currentPlayer.value = actor;
                    }
                    break;

                case 'pon':
                case 'chi':
                    if (p && event.consumed) {
                        event.consumed.forEach(t => removeTile(p, t));
                        const meld = [event.pai, ...event.consumed];
                        meld.type = 'pon';
                        p.fuuro.push(meld);
                        p.lastAction = type;
                        currentPlayer.value = actor;
                    }
                    break;

                case 'kakan':
                    if (p && event.pai) {
                        removeTile(p, event.pai);
                        // Find pon
                        const pon = p.fuuro.find(m => m.length === 3 && m.includes(event.pai) && m.type !== 'ankan');
                        if (pon) {
                            pon.push(event.pai);
                            pon.type = 'kakan';
                        }
                        p.lastAction = 'kakan';
                    }
                    break;
                // ... (rest of switch)

                case 'hora': // Win
                    if (event.deltas) {
                        event.deltas.forEach((d, i) => players.value[i].score += d);
                    }
                    if (p) {
                        p.agari = true;
                        p.lastAction = 'hora';
                    }
                    break;

                case 'ryukyoku':
                    if (event.deltas) {
                        event.deltas.forEach((d, i) => players.value[i].score += d);
                    }
                    break;

                case 'reach': // Riichi (Sichuan doesn't have it, but for compatibility)
                    if (p) p.riichi = true;
                    break;
            }
        };

        const arrowRotation = computed(() => {
            if (currentPlayer.value === null) return 0;
            // 0: Bottom (180deg? No, let's say 0 is Up/Top P2).
            // P0 (Bottom): 180deg
            // P1 (Right): 90deg
            // P2 (Top): 0deg
            // P3 (Left): -90deg

            // Wait, standard CSS rotation: 0 is usually Up or Right depending on icon.
            // Let's assume standard Arrow Icon points UP.

            switch (currentPlayer.value) {
                case 0: return 180; // Bottom
                case 1: return 90;  // Right
                case 2: return 0;   // Top
                case 3: return -90; // Left
            }
            return 0;
        });

        // ... helpers





        // --- Helpers ---
        const removeTile = (player, tile) => {
            const idx = player.tehai.lastIndexOf(tile); // Remove last instance (usually tsumo)
            if (idx >= 0) {
                player.tehai.splice(idx, 1);
            } else {
                // Fallback: simple remove
                const i = player.tehai.indexOf(tile);
                if (i >= 0) player.tehai.splice(i, 1);
            }
        };

        const sortHand = (player) => {
            // Sort standard order: m < p < s < z
            // Inside suit: 1 < 9
            const su = (t) => {
                const s = t.substr(1);
                if (s === 'm') return 1;
                if (s === 'p') return 2;
                if (s === 's') return 3;
                return 4; // z/ji
            };
            const val = (t) => {
                const v = parseInt(t[0]);
                return isNaN(v) ? 10 : v; // Honor tiles? E,S,W,N...
            };
            // Special map for honors if they use letters
            const honorMap = { 'E': 0, 'S': 1, 'W': 2, 'N': 3, 'P': 4, 'F': 5, 'C': 6 };

            const getOrder = (t) => {
                if (honorMap[t] !== undefined) return 400 + honorMap[t];

                const s = t.substr(1);
                const n = parseInt(t.substr(0, 1));

                let base = 0;
                if (s === 'm') base = 100;
                else if (s === 'p') base = 200;
                else if (s === 's') base = 300;

                return base + n;
            };

            player.tehai.sort((a, b) => getOrder(a) - getOrder(b));
        };

        // --- Playback Control ---
        const seekTo = (index) => {
            // Optimization: If seeking forward, process from current. 
            // If backward, full reset.
            if (index < currentEventIndex.value) {
                resetGameState();
                currentEventIndex.value = 0;
            }

            // Process forward
            while (currentEventIndex.value < index + 1 && currentEventIndex.value < totalEvents.value) {
                const ev = events.value[currentEventIndex.value];
                processEvent(ev);
                currentEventIndex.value++;
            }

            // Auto-scroll event list
            scrollToEvent(currentEventIndex.value - 1);
        };

        const nextEvent = () => {
            if (currentEventIndex.value < totalEvents.value) {
                const ev = events.value[currentEventIndex.value];
                processEvent(ev);
                currentEventIndex.value++;
                scrollToEvent(currentEventIndex.value - 1);
            } else {
                if (isPlaying.value) togglePlay(); // End of log
            }
        };

        const prevEvent = () => seekTo(currentEventIndex.value - 1);
        const firstEvent = () => seekTo(0);
        const lastEvent = () => seekTo(totalEvents.value);

        const togglePlay = () => {
            if (isPlaying.value) {
                clearInterval(playInterval);
                playInterval = null;
                isPlaying.value = false;
            } else {
                isPlaying.value = true;
                playStep();
            }
        };

        const playStep = () => {
            if (!isPlaying.value) return;
            nextEvent();
            if (currentEventIndex.value >= totalEvents.value) {
                togglePlay();
                return;
            }
            // Schedule next frame
            // Dynamic speed: 1.0x = 1000ms ?? No, that's too slow.
            // Let's say 1.0x = 500ms
            const delay = 800 / playbackSpeed.value;
            playInterval = setTimeout(playStep, delay);
        };

        const scrollToEvent = (idx) => {
            if (!eventListRef.value) return;
            // Simple logic: scroll so item is visible
            // const item = eventListRef.value.children[idx]; // Might be virtualized later
            // For now assume rendered:
            // This is tricky if items are reactive.
            // Let's scroll container.
            const el = eventListRef.value.querySelector('.current');
            if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        };

        // --- File Upload ---
        const fileInput = ref(null);
        const triggerUpload = () => fileInput.value.click();
        const handleFileUpload = async (event) => {
            const file = event.target.files[0];
            if (!file) return;
            uploadFile(file);
        };
        const handleDrop = async (event) => {
            const file = event.dataTransfer.files[0];
            if (!file) return;
            uploadFile(file);
        };
        const uploadFile = async (file) => {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.path) {
                    // Start loading
                    await refreshLogs();
                    // Try to find the new log and load it?
                    // Just load directly
                    loadLog(data.path);
                }
            } catch (e) {
                alert("Upload failed");
            }
        };

        // --- Utilities for Template ---
        const getTileText = (tile) => {
            if (!tile) return '';
            const map = {
                'E': '東', 'S': '南', 'W': '西', 'N': '北',
                'P': '白', 'F': '發', 'C': '中'
            };
            if (map[tile]) return map[tile];

            const num = tile[0];
            const suit = tile[1];
            const suitMap = { 'm': '萬', 'p': '筒', 's': '条' };
            return num + (suitMap[suit] || '');
        };

        const getTileClass = (tile) => {
            if (!tile) return '';
            if (tile.includes('m')) return 'man';
            if (tile.includes('p')) return 'pin';
            if (tile.includes('s')) return 'sou';
            return 'ji';
        };

        const getEventDetail = (e) => {
            if (e.type === 'tsumo') return `Draw ${getTileText(e.pai)}`;
            if (e.type === 'dahai') return `Discard ${getTileText(e.pai)}`;
            if (e.type === 'pon') return `Pon ${getTileText(e.pai)}`;
            if (e.type === 'ding_que') {
                const suitMap = { 'man': '萬', 'pin': '筒', 'sou': '条' };
                return `Ding Que: ${suitMap[e.suit] || e.suit}`;
            }
            if (e.type === 'hora') return `WIN! ${e.deltas ? e.deltas.join(',') : ''}`;
            return '';
        };

        const formatSize = (bytes) => {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        };

        // Init
        onMounted(() => {
            refreshLogs();
            // Start auto refresh
            setInterval(refreshLogs, 10000);
        });

        // Watch speed to update interval
        watch(playbackSpeed, () => {
            if (isPlaying.value) {
                // Restart loop with new speed
                clearTimeout(playInterval);
                playStep();
            }
        });

        return {
            searchQuery, filteredLogs, loadingLogs, currentLogPath,
            refreshLogs, loadLog, formatSize,
            triggerUpload, handleFileUpload, handleDrop, fileInput,

            gameLoaded, currentKyokuDisplay, tilesLeft, totalEvents, currentEventIndex,
            players, currentPlayer, lastDiscard, arrowRotation,

            showRightPanel, eventListRef, visibleEvents,

            getTileText, getTileClass,

            seekTo, nextEvent, prevEvent, firstEvent, lastEvent, togglePlay, isPlaying, playbackSpeed
        };
    }
};

createApp(App).mount('#app');
