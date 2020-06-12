<template>
  <div>
    <!-- <label>{{filter.name}}</label> -->
    <div @click="showConfusion()"
      class="filter__item filter__item--confusion filter__item--confusion__button"
      :class="{'filter__item--open' : confusionExpanded}"
    >
      <vega-lite
        @signal:pointer_tuple="updateSelection"
        @signal:hover_tuple="updateDatum"
        class="confusion-matrix"
        :autosize="autosize"
        :config="matrixConfig"
        :data="filter.values"
        :layer="matrixEncoding"
      />
    </div>
    <div
      v-if="confusionExpanded"
      v-click-outside="onClose"
      class="filter__item filter__item--confusion"
      v-bind:class="[
          {
            'expanded' :confusionExpanded,
            'expanded--full' : fullScreen
          }, filter.values.length > 10 ? 'hide-matrixText' : '']"
    >
      <re-button v-if="!fullScreen" @click="maximizeScreeen()" class="button-clear">
        <svgicon width="20" height="11" name="zoomin"></svgicon>Maximize
      </re-button>
      <re-button v-else @click="minimizeScreeen()" class="button-clear">
        <svgicon width="20" height="11" name="zoomout"></svgicon>Minimize
      </re-button>
      <div class="datum label-default">
        <svgicon
          class="close-filter-button"
          color="#787878"
          width="12"
          height="12"
          name="cross"
          @click="closeConfusionMatrix()"
        ></svgicon>
        <re-panel class="panel--inline" v-if="hoverActive">
          {{this.hover.count}} records
          <strong>{{this.hover.gold}}</strong> predicted as
          <strong>{{this.hover.predicted}}</strong>
        </re-panel>
      </div>
      <vega-lite
        @signal:pointer_tuple="updateSelection"
        @signal:hover_tuple="updateDatum"
        class="confusion-matrix"
        :autosize="autosize"
        :config="matrixConfig"
        :data="filter.values"
        :layer="matrixEncoding"
      />
      <div class="filter__buttons">
        <re-button
          class="button-tertiary--small button-tertiary--outline"
          @click="onClose()"
        >Cancel</re-button>
        <re-button
          :disabled="!this.selected.gold.length"
          class="button-primary--small"
          @click="onApplyConfusion()"
        >Apply</re-button>
      </div>
    </div>
    <div class="overlay" v-if="confusionExpanded"></div>
  </div>
</template>

<script>
import rePanel from '@/components/elements/core/rePanel/rePanel';
import reButton from '@/components/elements/core/reButton/reButton';
import '@/assets/iconsfont/cross';
import '@/assets/iconsfont/zoomin';
import '@/assets/iconsfont/zoomout';

export default {
  name: 'FilterConfusionMatrix',
  data: () => ({
    confusionExpanded: false,
    fullScreen: false,
    hoverActive: false,
    selected: {
      gold: '',
      predicted: '',
    },
    hover: {
      gold: '',
      predicted: '',
      count: '',
    },
    autosize: {
      type: 'none',
      resize: true,
    },
    matrixConfig: {
      view: {
        height: 360,
        width: 540,
        // height: 1600,
        // width: 1400,
      },
      axis: {
        labelLimit: 200,
        labelFontSize: 18,
        titleFontSize: 22,
        labelPadding: 50,
        titlePadding: 80,
      },
      mark: {
        color: '#8777D9',
      },
      scale: { bandPaddingInner: 0, bandPaddingOuter: 0 },
    },
    matrixEncoding: [{
      mark: 'rect',
      encoding: {
        y: {
          field: 'gold',
          type: 'nominal',
          scale: { rangeStep: null },
        },
        x: {
          field: 'predicted',
          type: 'nominal',
          scale: { rangeStep: null },
        },
        tooltip: { field: 'count', type: 'quantitative' },
        color: {
          condition: {
            selection: { not: 'pointer' },
            aggregate: 'sum',
            field: 'count',
            type: 'quantitative',
            scale: {
              range: ['#d6d3ea', '#8500FF', '#3B3269'],
              type: 'linear',
            },
          },
          value: '#4A4A4A',
        },
      },
      selection: {
        pointer: {
          type: 'single',
          on: 'click',
          fields: ['gold', 'predicted'],
        },
        hover: {
          type: 'single',
          on: 'rect:mouseover',
          fields: ['gold', 'predicted', 'count'],
        },
      },
      legend: null,
    }],
  }),
  props: ['filter'],
  components: {
    reButton,
    rePanel,
  },
  computed: {},
  methods: {
    updateSelection(tuple) {
      if (tuple && tuple.fields) {
        tuple.fields.forEach((key, i) => {
          this.selected[key] = tuple.values[i];
        });
      }
    },
    updateDatum(tuple) {
      tuple.fields.forEach((key, i) => {
        this.hover[key] = tuple.values[i];
      });
      this.hoverActive = true;
    },
    matrixShowLabel() {
      if (this.filter.values.length > 20) {
        this.autosize.type = 'none';
      } else {
        this.autosize.type = 'pad';
      }
    },
    showConfusion() {
      this.confusionExpanded = true;
      this.matrixShowLabel();
    },
    maximizeMatrixSize() {
      this.fullScreen = true;
      this.autosize.type = 'pad';
      this.matrixConfig.view.height = 1500;
      this.matrixConfig.view.width = 1600;
      this.matrixConfig.axis.labelLimit = 300;
      this.matrixConfig.axis.titlePadding = 250;
    },
    minimizeMatrixSize() {
      this.matrixShowLabel();
      this.fullScreen = false;
      this.matrixConfig.view.height = 360;
      this.matrixConfig.view.width = 540;
      this.matrixConfig.axis.labelLimit = 200;
      this.matrixConfig.axis.titlePadding = 80;
    },
    maximizeScreeen() {
      this.maximizeMatrixSize();
    },
    minimizeScreeen() {
      this.minimizeMatrixSize();
      this.autosize.type = 'none';
    },
    closeConfusionMatrix() {
      this.confusionExpanded = false;
      this.minimizeMatrixSize();
      this.autosize.type = 'none';
    },
    onApplyConfusion() {
      this.$emit('apply', this.selected);
      this.confusionExpanded = false;
      this.minimizeMatrixSize();
      this.autosize.type = 'none';
      this.selected.gold = '';
      this.selected.predicted = '';
    },
    onClose() {
      this.confusionExpanded = false;
      this.minimizeMatrixSize();
      this.autosize.type = 'none';
    },
  },
  updated() {
    if (this.confusionExpanded) {
      document.body.classList.add('--fixed');
    } else {
      document.body.classList.remove('--fixed');
    }
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/filter-matrix-confusion.scss";
@import "@/assets/scss/apps/classifier/filters.scss";

</style>
