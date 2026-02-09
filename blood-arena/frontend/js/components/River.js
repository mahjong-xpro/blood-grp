import Card from './Card.js';

export default {
    components: { Card },
    props: ['discards'],
    template: `
        <div class="river">
            <template v-for="(item, idx) in discards" :key="idx">
                <div class="tile">
                     <img :src="'/static/' + item + '.png'" :alt="item">
                </div>
            </template>
        </div>
    `
}
