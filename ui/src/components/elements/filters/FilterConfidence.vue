<template>
  <div>
    <!-- <label :class="{'--active' : visible}">{{filter.name}}</label> -->
    <div
      @click="expandConfidence"
      class="filter__item filter__item--confidence"
      :class="{'filter__item--open' : confidenceExpanded}"
    >
      <div class="confidence-content">
        <vega-lite
          class="confidence"
          :data="filter.values"
          :autosize="autosize"
          :config="config"
          :mark="mark"
          :encoding="encoding"
        />
      </div>
    </div>
    <div
      v-if="confidenceExpanded"
      v-click-outside="onClose"
      class="filter__item filter__item--confidence"
      :class="{expanded :confidenceExpanded}"
    >
      <div class="confidence-content">
        <re-panel class="panel--block range">{{min}}% to {{max}}%</re-panel>
        <vega-lite
          class="confidence"
          :data="filter.values"
          :autosize="autosize"
          :config="config"
          :mark="mark"
          :encoding="encoding"
        />
        <div class="range__container">
          <re-range
            ref="slider"
            v-if="confidenceExpanded"
            v-model="confidenceRanges"
            v-bind="rangeOptions"
          ></re-range>
        </div>
      </div>
      <div class="filter__buttons">
        <re-button
          class="button-tertiary--small button-tertiary--outline"
          @click="onClose()"
        >Cancel</re-button>
        <re-button
          class="button-primary--small"
          @click="onApplyConfidenceRange()"
        >Apply</re-button>
      </div>
    </div>
    <div class="overlay" v-if="confidenceExpanded"></div>
  </div>
</template>

<script>
import reButton from '@/components/elements/core/reButton/reButton';
import reRange from '@/components/elements/core/reRange/reRange';
import rePanel from '@/components/elements/core/rePanel/rePanel';

export default {
  name: 'FilterConfidence',
  props: ['filterName', 'data', 'filter'],
  data: () => ({
    confidenceExpanded: false,
    rangeOptions: {
      height: 4,
      dotSize: 20,
      min: 0,
      max: 100,
      interval: 1,
      show: true,
    },
    confidenceRanges: [],
    autosize: {
      type: 'none',
      resize: true,
      contains: 'padding',
    },
    mark: 'area',
    config: {
      mark: {
        color: '#D9D7E4',
        binSpacing: 0,
      },
      bar: {
        binSpacing: 0,
        discreteBandSize: 0,
        continuousBandSize: 0,
      },
      axis: {
        labels: false,
      },
      view: {
        height: 100,
        width: 400,
      },
    },
    encoding: {
      x: { field: 'key', type: 'ordinal', scale: { rangeStep: null } },
      y: { field: 'count', type: 'quantitative', aggregate: 'sum' },
    },
  }),
  methods: {
    expandConfidence() {
      this.confidenceExpanded = true;
    },
    onApplyConfidenceRange() {
      this.$emit('apply', this.confidenceRanges);
      this.confidenceExpanded = false;
    },
    onClose() {
      this.confidenceExpanded = false;
    },
  },
  components: {
    reButton,
    rePanel,
    reRange,
  },
  created() {
    this.confidenceRanges = [this.rangeOptions.min, this.rangeOptions.max];
  },
  computed: {
    visible() {
      return this.confidenceExpanded;
    },
    min() {
      return this.confidenceRanges[0];
    },
    max() {
      return this.confidenceRanges[1];
    },
  },
  updated() {
    if (this.confidenceExpanded) {
      document.body.classList.add('--fixed');
    } else {
      document.body.classList.remove('--fixed');
    }
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/filters.scss";
@import "@/assets/scss/apps/classifier/filter-confidence.scss";
</style>
