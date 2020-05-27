<template>
  <span class="token__container">
    <span
      :class="tokenItemType !== 'list' ? 'token__group--list' : ''"
      class="token__group"
      v-for="token in tokens()"
      :key="token.index"
    >
      <div v-for="tokenItem in token" :key="tokenItem.index" :class="['atom', Math.sign(tokenItem.grad) !== 1 ? `grad-neg-${tokenItem.percent}` : `grad-${tokenItem.percent}`, {'token--margin' : tokenItem.token.length > 1}]">
        <span v-if="tokenItem.grad" class="atom__tooltip">{{tokenItem.grad}}</span>
        <span v-html="tokenItem.token">{{tokenItem.token === ' ' ? '&nbsp;' : tokenItem.token}}</span>
      </div>
    </span>
  </span>
</template>

<script>
export default {
  name: 'explain',
  data: () => ({
    tokenItemType: undefined,
  }),
  methods: {
    tokens() {
      let list = [];
      return this.interpret.map((interpretItem) => {
        if (!Array.isArray(interpretItem)) {
          list = [interpretItem];
        } else {
          list = interpretItem;
          this.tokenItemType = 'list';
        }
        return list.map((input) => {
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
      });
    },
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
    predictedOk: {
      type: Boolean,
      default: true,
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/explain.scss";
.token {
  &__container {
    display: inline-block;
    vertical-align: top;
  }
  &__group {
    display: block;
    margin-bottom: 0.5em;
    &--list {
      display: inline-block;
    }
  }
  &--margin {
    margin-left: 0.25em;
    margin-right: 0.25em;
  }
}

</style>
