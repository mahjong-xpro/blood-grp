import PlayerHand from './PlayerHand.js';
import River from './River.js';
import InfoPanel from './InfoPanel.js';
import AnalysisPanel from './AnalysisPanel.js';

export default {
    components: { PlayerHand, River, InfoPanel, AnalysisPanel },
    props: ['state', 'analysis'], // 'state' is the global game state object
    emits: ['action'],

    setup(props, { emit }) {
        const getPlayer = (offset) => {
            // offset 0 = me, 1 = right, 2 = opposite, 3 = left
            // playerIds are 0-3. myPlayerId is in state.
            const myId = props.state.myPlayerId;
            const targetId = (myId + offset) % 4;
            return {
                id: targetId,
                score: props.state.scores[targetId],
                isTurn: props.state.currentActor === targetId,
                agari: props.state.agari && props.state.agari[targetId]
            };
        };

        const getHand = (offset) => {
            // For opponents, we just show back of cards or simple count
            // But 'state.tehai' only has MY tiles.
            // Opponent hands are hidden.
            // We can visualize by count if available in state.
            const myId = props.state.myPlayerId;
            const targetId = (myId + offset) % 4;
            if (offset === 0) return props.state.tehai;

            // Mock opponent hand (just tile count 13/10 etc)
            // We need to track tile counts.
            // For now, return array of "back" tiles based on arbitrary count 13.
            // Ideally state should track hand counts.
            return Array(13).fill('back');
        };

        const getDiscards = (offset) => {
            const myId = props.state.myPlayerId;
            const targetId = (myId + offset) % 4;
            return props.state.discards[targetId] || [];
        };

        const handleDiscard = (tile) => {
            emit('action', { type: 'dahai', pai: tile });
        };

        return {
            getPlayer,
            getHand,
            getDiscards,
            handleDiscard
        };
    },
    template: `
        <div class="game-board">
            
            <!-- Analysis Overlay -->
            <AnalysisPanel :analysis="analysis" />
        
            <!-- Top Player (Opposite) -->
            <div class="player-area player-top">
                <InfoPanel :player="getPlayer(2)" />
                <div class="hand opponent-hand">
                     <div class="tile back" v-for="n in 13"></div>
                </div>
                <River :discards="getDiscards(2)" />
            </div>

            <!-- Left Player -->
            <div class="player-area player-left">
                <InfoPanel :player="getPlayer(3)" />
                <div class="hand opponent-hand">
                     <div class="tile back" v-for="n in 13"></div>
                </div>
                <River :discards="getDiscards(3)" />
            </div>

            <!-- Right Player -->
            <div class="player-area player-right">
                <InfoPanel :player="getPlayer(1)" />
                <div class="hand opponent-hand">
                     <div class="tile back" v-for="n in 13"></div>
                </div>
                <River :discards="getDiscards(1)" />
            </div>

            <!-- Bottom Player (Me) -->
            <div class="player-area player-bottom">
                <River :discards="getDiscards(0)" />
                <PlayerHand 
                    :tiles="getHand(0)" 
                    :isMyTurn="getPlayer(0).isTurn"
                    :analysis="analysis"
                    @discard="handleDiscard" />
                <InfoPanel :player="getPlayer(0)" />
            </div>
            
            <!-- Center Info -->
            <div class="center-info">
                 <div class="turn-indicator" :class="{ active: getPlayer(0).isTurn }">
                     {{ getPlayer(0).isTurn ? "YOUR TURN" : "WAITING..." }}
                 </div>
                 <div style="color: #666">Tiles Left: {{ state.tilesLeft }}</div>
            </div>
            
        </div>
    `
}
