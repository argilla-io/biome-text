
<template>
  <form @submit.prevent="$emit('submit', query)">
    <div :class="{'active' : query.length, 'expanded' : expand}" v-click-outside="collapse">
      <re-input-container class="searchbar">
      <re-input ref="input" class="searchbar__input" placeholder="Search records" v-model="query"></re-input>
      <svgicon
        class="searchbar__button"
        name="search"
        width="20"
        height="40"
        @click="submitQuery()"
      ></svgicon>
      <svgicon
        v-show="!expand"
        class="searchbar__button--expand"
        name="search"
        width="20"
        height="40"
        @click="showExpanded()"
      ></svgicon>
      </re-input-container>
    </div>
  </form>
</template>

<script>
import '@/assets/iconsfont/search';
import reInputContainer from '@/components/elements/core/reInputContainer/reInputContainer';
import reInput from '@/components/elements/core/reInputContainer/reInput';

export default {
  name: 'searchbar',
  data: () => ({
    loading: false,
    expand: false,
  }),
  props: ['query'],
  methods: {
    onCloseInput() {
      this.query = '';
      this.$emit('clear');
    },
    showExpanded() {
      this.expand = true;
      this.$refs.input.$el.focus();
    },
    collapse() {
      if (this.expand === true) {
        this.expand = false;
      }
    },
    submitQuery() {
      if (this.query.length) {
        this.$emit('submit', this.query)
      } else {
        this.expand = false;
      }
    }
  },
  components: {
    reInputContainer,
    reInput,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/search-bar.scss";
.searchbar {
  margin-bottom: 0;
  margin-top: 0;
}
</style>
