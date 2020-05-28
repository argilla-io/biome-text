<template>
  <span>
    <span
      :class="['atom', Math.sign(token.grad) !== 1 ? `grad-neg-${token.percent}` : `grad-${token.percent}`, {'token-margin' : token.token.length > 1}]"
      v-for="token in tokens"
      :key="token.index"
    >
      <span v-if="token.grad" class="atom__tooltip">{{token.grad}}</span>
      <span v-html="token.token">{{token.token === ' ' ? '&nbsp;' : token.token}}</span>
    </span>
  </span>
</template>

<script>
export default {
  name: 'interpretations',
  computed: {
    tokens() {
      let interpretList = [];
      this.interpret.forEach((interpretItem) => {
        if (Array.isArray(interpretItem)) {
          return interpretItem.forEach((i) => {
            interpretList.push(i);
          });
        }
        interpretList = this.interpret;
        return interpretList;
      });
      return interpretList.map((input) => {
        const grad = input.grad.toFixed(3);
        let percent = Math.round(Math.abs(grad) * 100);
        if (percent !== 0) {
          /* eslint-disable no-mixed-operators */
          const p = 1.5; // color sensitivity (values from 1 to 4)
          const s = 100 / Math.log10(100) ** p;
          percent = Math.round(Math.log10(percent) ** p * s);
        }

        return {
          token: this.highlightSearch(input.token, this.query),
          percent: percent.toString(),
          grad,
        };
      });
    },
  },
  methods: {
    highlightSearch(option, searchText) {
      if (!searchText) {
        return option;
      }
      return option.toString().replace(new RegExp(searchText, 'i'), match => `<span class="highlight-text">${match}</span>`);
    },
  },
  props: {
    interpret: {
      type: Array,
    },
    query: {
      type: String,
      default: undefined,
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/interpretations.scss";
.token-margin {
  margin-left: 0.25em;
  margin-right: 0.25em;
}
</style>
