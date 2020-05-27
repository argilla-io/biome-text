<template>
  <div>
      <div ref="scroll" class="record__scroll" :class="{'record__scroll--prevent' : !allowScroll}" v-if="showEntityClassifier">
        <re-button class="record__scroll__button button-icon" @click="allowScroll = !allowScroll">
          <svgicon :name="allowScroll ? 'unlock' : 'lock'" width="15" height="14"></svgicon>
        </re-button>
        <span class="record__textual">
          <span v-html="textualRecord(recordValue)">{{textualRecord(recordValue)}}</span>
        </span>
      </div>
      <span v-else>
        <span v-if="Object.keys(recordValue).length > 1" class="record__key">{{index}}: </span>
        <span :class="{'record__value--highlighted' : index === 'tokens.inputs.value'}">
          <span v-html="highlightSearch(recordValue, this.query)"></span>
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
  name: 'record-text',
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
      type: [String, Number],
    },
    query: {
      type: String,
      default: undefined,
    },
  },
  methods: {
    textualRecord(recordValue) {
      if (this.record['tokens.inputs.span_start']) {
        /* eslint no-extend-native: ["error", { "exceptions": ["String"] }] */
        String.prototype.splice = function textualPosition(idx, rem, str) {
          return this.slice(0, idx) + str + this.slice(idx + Math.abs(rem));
        };
        const r = this.record['tokens.inputs.span_start'] - this.record['tokens.inputs.span_end'];
        const result = recordValue.splice(this.record['tokens.inputs.span_start'], r, `<span class='record__text__value--highlighted'>${this.record['tokens.inputs.value']}</span>`);
        return (function textualPosition() {
          return result;
        }());
      }
      return this.record;
    },
    highlightSearch(option, searchText) {
      if (!searchText) {
        return option;
      }
      return option.toString().replace(new RegExp(searchText, 'i'), match => `<span class="highlight-text">${match}</span>`);
    },
  },
  mounted() {
    if (this.$refs.scroll && this.record['tokens.inputs.value']) {
      setTimeout(() => {
        this.$nextTick(() => {
          const item = this.$el.querySelector('.record__text__value--highlighted');
          const table = this.$refs.scroll;
          const top = item.offsetTop;
          table.scrollTop = top - 100;
        });
      }, 100);
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
