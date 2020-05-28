<template>
  <div class="metadata">
    <p class="metadata__title" :title="name" :key="name.index" v-for="name in inputs">
      <span v-if="!Array.isArray(name)">{{name | truncate(100)}}</span>
    </p>
    <div class="metadata__container">
      <div class="metadata__blocks" v-for="(row, index) in Object.entries(previewData)" :key="index">
        <div :class="['metadata__block', {'--selected' : metaFilterSelected(row)}]" :ref="row[0]">
          <div class="metadata__block__item" v-for="(col, i) in row" :key="i">{{col}}</div>
        </div>
          <re-button v-if="metaFiltrable"
            class="metadata__block__button button-clear"
            @click="addFilter(row)"
          >Filter</re-button>
      </div>
    </div>
    <div class="metadata__buttons">
      <re-button
        class="button-tertiary--small button-tertiary--outline"
        @click="$emit('cancel')"
      >Cancel</re-button>
      <re-button
        :disabled="disableButton"
        class="button-primary--small"
        @click="applySelectedFilters()"
      >Apply</re-button>
    </div>
  </div>
</template>
<script>
import reButton from '@/components/elements/core/reButton/reButton';

export default {
  data: () => ({
    selectedKeys: [],
    selectedFilters: [],
    disableButton: true,
  }),
  name: 'metadata',
  props: ['previewData', 'metaFiltrable', 'filtersStatus', 'inputs'],
  filters: {
    truncate(string, value) {
      if (string.length > value) {
        return `${string.substring(0, value)}...`;
      }
      return string;
    },
  },
  methods: {
    addFilter(f) {
      const metafilter = {
        key: f[0],
        value: f[1],
      };
      const filterRef = this.$refs[metafilter.key][0];
      if (filterRef.classList.contains('--selected')) {
        const filterIndex = this.selectedFilters.map(s => s.value).indexOf(metafilter.value);
        this.selectedFilters.splice(filterIndex, 1);
        filterRef.classList.remove('--selected');
      } else {
        this.selectedFilters.push(metafilter);
        filterRef.classList.add('--selected');
      }
      this.disableButton = false;
    },
    applySelectedFilters() {
      this.$emit('metafilterapply', this.selectedFilters);
      this.disableButton = true;
    },
    metaFilterSelected(metaFilter) {
      let isSelected = false;
      if (this.filtersStatus[metaFilter[0]]) {
        this.filtersStatus[metaFilter[0]].forEach((f) => {
          isSelected = metaFilter[1].toString() === f.toString();
        });
      }
      return isSelected;
    },
    appliedFilters() {
      const metaKeys = [];
      Object.keys(this.previewData).forEach((mk) => {
        metaKeys.push(mk);
      });
      const selected = Object.keys(this.filtersStatus).filter(f => metaKeys.includes(f));
      this.selectedKeys = selected;
      const selectedF = selected.map((s) => {
        const sf = {
          key: s,
          value: this.filtersStatus[s][0],
        };
        return sf;
      });
      this.selectedFilters = selectedF;
    },
  },
  mounted() {
    this.appliedFilters();
  },
  components: {
    reButton,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/metadata.scss";
</style>
