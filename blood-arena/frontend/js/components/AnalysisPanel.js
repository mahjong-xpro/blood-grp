export default {
    props: ['analysis'],
    computed: {
        winRate() {
            // Estimate win rate from best Q-value
            // Mortal Q-values are roughly centered around 0? No, they seem to be normalized scores?
            // Actually, in Mortal v4, Q-values are expected rewards.
            // Let's just show raw Q-value bar for now, scaled.
            // Assuming Q is roughly [-1, 1] or similar range?
            // "candidates": [{"q": 0.123, ...}]
            if (!this.analysis || !this.analysis.candidates || this.analysis.candidates.length === 0) return 0;
            const q = this.analysis.candidates[0].q;
            // Map [-2, 2] to [0, 100]? Or just use it as is if it's probability.
            // If it's probability (0-1), then *100.
            // Let's assume it's probability-like for visualization.
            return Math.min(100, Math.max(0, q * 100)).toFixed(1);
        },
        candidates() {
            return this.analysis ? this.analysis.candidates : [];
        }
    },
    template: `
        <div class="analysis-panel" v-if="analysis && analysis.candidates">
            <div class="analysis-header">
                <span class="analysis-title">AI ANALYSIS</span>
                <span>{{ winRate }}%</span>
            </div>
            <div class="win-rate-bar">
                <div class="win-rate-fill" :style="{ width: winRate + '%' }"></div>
            </div>
            
            <ul class="candidate-list">
                <li v-for="c in candidates" :key="c.idx" class="candidate-item">
                    <div class="candidate-tile">
                        <span v-if="c.type === 'discard'">Discard</span>
                        <span v-else>Action {{ c.idx }}</span>
                        
                        <div v-if="c.tile" class="mini-tile">
                             <img :src="'/static/' + c.tile + '.png'" style="width:100%; height:100%">
                        </div>
                    </div>
                    <span class="candidate-score">{{ c.q.toFixed(4) }}</span>
                </li>
            </ul>
        </div>
    `
}
