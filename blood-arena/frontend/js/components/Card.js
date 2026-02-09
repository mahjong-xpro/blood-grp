export default {
    props: ['tile', 'selected', 'recommended', 'bad', 'isTurn'],
    emits: ['click'],
    template: `
        <div class="tile" 
             :class="{ 'selected': selected, 'discard-recommended': recommended, 'discard-bad': bad }"
             @click="$emit('click')">
            <img :src="'/static/' + tile + '.png'" :alt="tile">
        </div>
    `
}
