import { createApp, ref, computed, onMounted, watch } from 'https://unpkg.com/vue@3/dist/vue.esm-browser.js';

// --- Tile helpers (v2: integer IDs 0-26) ---
const TILE_NAMES = [
    'Man1','Man2','Man3','Man4','Man5','Man6','Man7','Man8','Man9',
    'Pin1','Pin2','Pin3','Pin4','Pin5','Pin6','Pin7','Pin8','Pin9',
    'Sou1','Sou2','Sou3','Sou4','Sou5','Sou6','Sou7','Sou8','Sou9',
];
const SUIT_CHARS = ['m', 'p', 's'];

const tileToImg = (id) => {
    if (id === undefined || id === null || id < 0 || id > 26) return '/static/images/tiles/Blank.png';
    return `/static/images/tiles/${TILE_NAMES[id]}.png`;
};
const tileToAudio = (id) => {
    if (id === undefined || id === null) return null;
    const suit = SUIT_CHARS[Math.floor(id / 9)];
    const rank = (id % 9) + 1;
    return `${rank}${suit}.m4a`;
};
const tileToClass = (id) => {
    if (id === undefined || id === null) return '';
    const suit = Math.floor(id / 9);
    return ['man', 'pin', 'sou'][suit] || '';
};
const tileToText = (id) => {
    if (id === undefined || id === null) return '';
    const suit = ['萬', '筒', '条'][Math.floor(id / 9)];
    const rank = (id % 9) + 1;
    return `${rank}${suit}`;
};

const App = {
    setup() {
        // --- State ---
        const searchQuery = ref('');
        const logList = ref([]);
        const loadingLogs = ref(false);
        const currentLogPath = ref(null);

        const gameLoaded = ref(false);
        const currentEventIndex = ref(0);
        const events = ref([]);
        const totalEvents = computed(() => events.value.length);
        const gameInfo = ref({});

        const isPlaying = ref(false);
        const playbackSpeed = ref(1.0);
        let playInterval = null;

        const players = ref([
            { name: 'Player 0', score: 100000, dingque: null, dingque_suit: '', dingque_char: '', agari: false, riichi: false, tehai: [], kawa: [], fuuro: [], lastAction: '' },
            { name: 'Player 1', score: 100000, dingque: null, dingque_suit: '', dingque_char: '', agari: false, riichi: false, tehai: [], kawa: [], fuuro: [], lastAction: '' },
            { name: 'Player 2', score: 100000, dingque: null, dingque_suit: '', dingque_char: '', agari: false, riichi: false, tehai: [], kawa: [], fuuro: [], lastAction: '' },
            { name: 'Player 3', score: 100000, dingque: null, dingque_suit: '', dingque_char: '', agari: false, riichi: false, tehai: [], kawa: [], fuuro: [], lastAction: '' },
        ]);
        const tilesLeft = ref(55);
        const currentPlayer = ref(null);
        const lastDiscard = ref(null);

        const showRightPanel = ref(true);
        const eventListRef = ref(null);
        const pov = ref(0);

        // --- Computed ---
        const filteredLogs = computed(() => {
            if (!searchQuery.value) return logList.value;
            const q = searchQuery.value.toLowerCase();
            return logList.value.filter(l => l.name.toLowerCase().includes(q));
        });

        const visibleEvents = computed(() =>
            events.value.map((e, i) => ({
                index: i,
                type: e.type,
                actor: e.player !== undefined ? e.player : e.actor,
                detail: getEventDetail(e),
            }))
        );

        const getVisualPosition = (playerIdx) => (playerIdx - pov.value + 4) % 4;

        const arrowRotation = computed(() => {
            if (currentPlayer.value === null) return 0;
            const pos = getVisualPosition(currentPlayer.value);
            return [180, 90, 0, -90][pos] ?? 0;
        });

        // --- Data Loading ---
        const refreshLogs = async () => {
            loadingLogs.value = true;
            try {
                const res = await fetch('/api/logs');
                const data = await res.json();
                if (data.logs) logList.value = data.logs;
            } catch (e) {
                console.error('Failed to load logs', e);
            } finally {
                loadingLogs.value = false;
            }
        };

        const loadLog = async (path) => {
            if (isPlaying.value) togglePlay();
            const encodedPath = path.split('/').map(encodeURIComponent).join('/');
            try {
                const res = await fetch(`/api/log/${encodedPath}`);
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                events.value = data.events || [];
                gameInfo.value = data;
                gameLoaded.value = true;
                currentLogPath.value = path;
                pov.value = data.hero_index ?? 0;
                if (data.names) gameInfo.value.playerNames = data.names;
                seekTo(0);
            } catch (e) {
                alert(`Error loading log: ${e.message}`);
            }
        };

        // --- Game Logic ---
        const resetGameState = () => {
            const names = gameInfo.value.playerNames || ['Player 0', 'Player 1', 'Player 2', 'Player 3'];
            players.value = names.map(name => ({
                name,
                score: 100000,
                dingque: null, dingque_suit: '', dingque_char: '',
                agari: false, riichi: false,
                tehai: [], kawa: [], fuuro: [], lastAction: '',
            }));
            tilesLeft.value = 55;
            currentPlayer.value = null;
            lastDiscard.value = null;
        };

        // --- Audio ---
        const playSound = (filename) => {
            const audio = new Audio(`/static/audio/${filename}`);
            audio.volume = 1.0;
            audio.play().catch(() => {});
        };
        const testAudio = () => playSound('pon.m4a');

        // --- Event Processing (v2 event types) ---
        const processEvent = (event, silent = false) => {
            const type = event.type;
            const pid = event.player !== undefined ? event.player : event.actor;
            const p = pid !== undefined ? players.value[pid] : null;

            switch (type) {
                case 'game_start':
                    if (event.names) event.names.forEach((n, i) => players.value[i].name = n);
                    if (event.initial_scores) event.initial_scores.forEach((s, i) => players.value[i].score = s);
                    break;

                case 'ding_que': {
                    if (!p) break;
                    const suitMap = { man: '萬', pin: '筒', sou: '条' };
                    p.dingque = event.suit;
                    p.dingque_char = suitMap[event.suit] || '?';
                    p.dingque_suit = event.suit || '';
                    if (!silent) playSound('dingque.m4a');
                    break;
                }

                case 'draw':
                    if (!p) break;
                    p.tehai.push(event.tile);
                    p.lastAction = 'draw';
                    currentPlayer.value = pid;
                    if (tilesLeft.value > 0) tilesLeft.value--;
                    break;

                case 'discard':
                    if (!p) break;
                    removeTile(p, event.tile);
                    p.kawa.push(event.tile);
                    sortHand(p);
                    p.lastAction = 'discard';
                    currentPlayer.value = pid;
                    lastDiscard.value = { actor: pid, tile: event.tile };
                    if (!silent) {
                        const af = tileToAudio(event.tile);
                        if (af) playSound(af);
                    }
                    break;

                case 'pon': {
                    if (!p) break;
                    removeTile(p, event.tile);
                    removeTile(p, event.tile);
                    const ponMeld = [event.tile, event.tile, event.tile];
                    ponMeld.type = 'pon';
                    ponMeld.from = event.from;
                    p.fuuro.push(ponMeld);
                    p.lastAction = 'pon';
                    currentPlayer.value = pid;
                    if (!silent) playSound('pon.m4a');
                    break;
                }

                case 'min_kan': {
                    if (!p) break;
                    removeTile(p, event.tile);
                    removeTile(p, event.tile);
                    removeTile(p, event.tile);
                    const minkMeld = [event.tile, event.tile, event.tile, event.tile];
                    minkMeld.type = 'minkan';
                    minkMeld.from = event.from;
                    p.fuuro.push(minkMeld);
                    p.lastAction = 'minkan';
                    currentPlayer.value = pid;
                    if (!silent) playSound('kan.m4a');
                    break;
                }

                case 'an_kan': {
                    if (!p) break;
                    for (let i = 0; i < 4; i++) removeTile(p, event.tile);
                    const ankMeld = [event.tile, event.tile, event.tile, event.tile];
                    ankMeld.type = 'ankan';
                    p.fuuro.push(ankMeld);
                    p.lastAction = 'ankan';
                    if (!silent) playSound('kan.m4a');
                    break;
                }

                case 'ka_kan': {
                    if (!p) break;
                    removeTile(p, event.tile);
                    const pon = p.fuuro.find(m => m.type === 'pon' && m[0] === event.tile);
                    if (pon) { pon.push(event.tile); pon.type = 'kakan'; }
                    p.lastAction = 'kakan';
                    if (!silent) playSound('kan.m4a');
                    break;
                }

                case 'kan_payment':
                    players.value[event.payer].score -= event.amount;
                    players.value[event.receiver].score += event.amount;
                    break;

                case 'tsumo': // win by self-draw
                    if (!p) break;
                    p.agari = true;
                    p.lastAction = 'tsumo';
                    if (event.scores) event.scores.forEach((s, i) => players.value[i].score = s);
                    if (!silent) playSound('tsumo.m4a');
                    break;

                case 'ron': // win by discard
                    if (!p) break;
                    p.agari = true;
                    p.lastAction = 'ron';
                    if (event.scores) event.scores.forEach((s, i) => players.value[i].score = s);
                    if (!silent) playSound('ron.m4a');
                    break;

                case 'game_end':
                    if (event.final_scores) event.final_scores.forEach((s, i) => players.value[i].score = s);
                    break;
            }
        };

        // --- Helpers ---
        const removeTile = (player, tile) => {
            // v2: tiles are integers, compare with ===
            let idx = -1;
            for (let i = player.tehai.length - 1; i >= 0; i--) {
                if (player.tehai[i] === tile) { idx = i; break; }
            }
            if (idx >= 0) player.tehai.splice(idx, 1);
        };

        const sortHand = (player) => {
            player.tehai.sort((a, b) => a - b);
        };

        // --- Playback ---
        const seekTo = (index) => {
            if (index < currentEventIndex.value) {
                resetGameState();
                currentEventIndex.value = 0;
            }
            while (currentEventIndex.value < index && currentEventIndex.value < totalEvents.value) {
                processEvent(events.value[currentEventIndex.value], true);
                currentEventIndex.value++;
            }
            scrollToEvent(currentEventIndex.value - 1);
        };

        const nextEvent = () => {
            if (currentEventIndex.value < totalEvents.value) {
                processEvent(events.value[currentEventIndex.value]);
                currentEventIndex.value++;
                scrollToEvent(currentEventIndex.value - 1);
            } else {
                if (isPlaying.value) togglePlay();
            }
        };

        const prevEvent = () => seekTo(currentEventIndex.value - 1);
        const firstEvent = () => seekTo(0);
        const lastEvent = () => seekTo(totalEvents.value);

        const togglePlay = () => {
            if (isPlaying.value) {
                clearTimeout(playInterval);
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
            if (currentEventIndex.value >= totalEvents.value) { togglePlay(); return; }
            playInterval = setTimeout(playStep, 800 / playbackSpeed.value);
        };

        const scrollToEvent = () => {
            if (!eventListRef.value) return;
            const el = eventListRef.value.querySelector('.current');
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        };

        // --- File Upload ---
        const fileInput = ref(null);
        const triggerUpload = () => fileInput.value.click();
        const handleFileUpload = (event) => {
            const file = event.target.files[0];
            if (file) uploadFile(file);
        };
        const handleDrop = (event) => {
            const file = event.dataTransfer.files[0];
            if (file) uploadFile(file);
        };
        const uploadFile = async (file) => {
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.path) { await refreshLogs(); loadLog(data.path); }
            } catch (e) { alert('Upload failed'); }
        };

        // --- Template Utilities ---
        const getTileImage = (id) => tileToImg(id);
        const getTileClass = (id) => tileToClass(id);

        const getEventDetail = (e) => {
            const t = e.type;
            if (t === 'draw') return `摸 ${tileToText(e.tile)}`;
            if (t === 'discard') return `打 ${tileToText(e.tile)}`;
            if (t === 'pon') return `碰 ${tileToText(e.tile)}`;
            if (t === 'min_kan') return `明杠 ${tileToText(e.tile)}`;
            if (t === 'an_kan') return `暗杠 ${tileToText(e.tile)}`;
            if (t === 'ka_kan') return `加杠 ${tileToText(e.tile)}`;
            if (t === 'tsumo') return `自摸 ${tileToText(e.tile)}`;
            if (t === 'ron') return `荣 ${tileToText(e.tile)}`;
            if (t === 'ding_que') { const m = { man: '萬', pin: '筒', sou: '条' }; return `定缺 ${m[e.suit] || e.suit}`; }
            if (t === 'kan_payment') return `杠费 ${e.amount}`;
            if (t === 'game_end') return `终局`;
            return '';
        };

        const formatSize = (bytes) => {
            if (!bytes) return '0 B';
            const k = 1024, sizes = ['B', 'KB', 'MB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        };

        onMounted(() => {
            refreshLogs();
            setInterval(refreshLogs, 10000);
        });

        watch(playbackSpeed, () => {
            if (isPlaying.value) { clearTimeout(playInterval); playStep(); }
        });

        return {
            searchQuery, filteredLogs, loadingLogs, currentLogPath,
            refreshLogs, loadLog, formatSize,
            triggerUpload, handleFileUpload, handleDrop, fileInput,
            gameLoaded, tilesLeft, totalEvents, currentEventIndex,
            players, currentPlayer, lastDiscard, arrowRotation,
            showRightPanel, eventListRef, visibleEvents,
            getTileImage, getTileClass,
            seekTo, nextEvent, prevEvent, firstEvent, lastEvent,
            togglePlay, isPlaying, playbackSpeed,
            pov, getVisualPosition, testAudio,
        };
    }
};

createApp(App).mount('#app');
