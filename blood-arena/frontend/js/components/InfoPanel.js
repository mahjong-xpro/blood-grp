export default {
    props: ['player'],
    template: `
        <div class="score-panel" :class="{ active: player.isTurn }">
            <div>Player {{ player.id }}</div>
            <div style="font-size: 1.2em; color: #fff">{{ player.score }}</div>
            <div v-if="player.agari" style="color: #ff1744">WIN</div>
        </div>
    `
}
