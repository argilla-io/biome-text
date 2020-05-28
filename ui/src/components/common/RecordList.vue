<template>
  <div>
    <span>
      <span v-if="!showEntityClassifier" class="record__key">{{index}}: </span>
      <span ref="list" class="record__list record__scroll" :class="{'record__scroll--prevent' : !allowScroll}">
        <re-button class="record__scroll__button button-icon" @click="allowScroll = !allowScroll">
          <svgicon :name="allowScroll ? 'lock' : 'unlock'" width="15" height="14"></svgicon>
        </re-button>
        <span v-for="(value, vIndex) in excludeEmptyRecord(recordValue)" :key="vIndex">
          <span class="record__list__value" :ref="valueHasPosition(vIndex) ? 'highlighted': ''" :class="{'record__list__value--highlighted' : valueHasPosition(vIndex)}">
            <span v-html="highlightSearch(value, query)">{{value}}</span>
          </span>
        </span>
      </span>
    </span>
  </div>
</template>
<script>
import '@/assets/iconsfont/lock';
import '@/assets/iconsfont/unlock';
import reButton from '@/components/elements/core/reButton/reButton';

export default {
  data: () => ({
    allowScroll: false,
  }),
  name: 'record-list',
  props: {
    showEntityClassifier: {
      type: Boolean,
      default: false,
    },
    record: {
      type: Object,
    },
    index: {
      type: String,
    },
    recordValue: {
      type: Array,
    },
    query: {
      type: String,
      default: undefined,
    },
  },
  methods: {
    excludeEmptyRecord(recordValue) {
      return recordValue.filter(rec => rec !== '');
    },
    valueHasPosition(position) {
      return position === this.record['tokens.position'];
    },
    highlightSearch(option, searchText) {
      if (!searchText) {
        return option;
      }
      return option.toString().replace(new RegExp(searchText, 'i'), match => `<span class="highlight-text">${match}</span>`);
    },
  },
  mounted() {
    if (this.$refs.highlighted) {
      setTimeout(() => {
        this.$nextTick(() => {
          const item = this.$refs.highlighted[0];
          const table = this.$refs.list;
          const top = item.offsetTop;
          table.scrollTop = top - 80;
        });
      }, 1000);
    }
  },
  components: {
    reButton,
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/record.scss";
</style>
