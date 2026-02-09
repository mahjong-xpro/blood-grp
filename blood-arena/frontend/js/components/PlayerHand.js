import Card from './Card.js';

export default {
    components: { Card },
    props: ['tiles', 'isMyTurn', 'analysis'],
    emits: ['discard'],
    data() {
        return {
            selectedTileIdx: -1
        };
    },
    methods: {
        getTileStatus(tile, idx) {
            if (!this.analysis || !this.analysis.candidates) return {};

            // Map tile to analysis candidates
            // Analysis candidates refer to action index. 
            // Discard action indices are 0-26 (Simple2DArray encoding)
            // But 'actions' from Mortal are integers.
            // We need to match tile string "1m" to ID.

            // Simplified: If backend sends clear recommendations like "best_action" = 3 (1m)
            // We just highlight the best action.

            // For now, let's look at best_action in analysis
            if (this.analysis.best_action) {
                // best_action usually has "type": "dahai", "pai": "1m"
                if (this.analysis.best_action.pai === tile) {
                    return { recommended: true };
                }
            }
            return {};
        },
        handleClick(tile, idx) {
            if (!this.isMyTurn) return;

            if (this.selectedTileIdx === idx) {
                // Confirm discard
                this.$emit('discard', tile);
                this.selectedTileIdx = -1;
            } else {
                // Select
                this.selectedTileIdx = idx;
            }
        }
    },
    template: `
        <div class="hand">
            <Card v-for="(tile, idx) in tiles" 
                  :key="idx" 
                  :tile="tile"
                  :selected="selectedTileIdx === idx"
                  :recommended="getTileStatus(tile, idx).recommended"
                  :bad="getTileStatus(tile, idx).bad"
                  @click="handleClick(tile, idx)" />
        </div>
    `
}
