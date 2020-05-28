<template>
  <aside class="sidebar explore">
    <transition name="fade" appear>
      <div class="sidebar__wrapper">
        <div class="sidebar__content">
          <div :class="['progress__block', loadingQ ? 'loading-skeleton' : '']">
            <p>Metrics %</p>
            <span v-for="(metric,key) in metricsOverall" :key="key">
              <label>{{metric.key}}</label>
              <span class="progress">{{metric.value}}</span>
              <re-progress
                :class="metric.key"
                re-mode="determinate"
                :progress="parseInt(metric.value)"
              ></re-progress>
            </span>
          </div>
        </div>
        <div class="sidebar__content" v-for="(entries,key) in metrics" :key="key">
          <div :class="['progress__block', loadingQ ? 'loading-skeleton' : '']">
            <collapsable-progress-bars :entries="entries" :entriesKey="key" :barsNumber="3"></collapsable-progress-bars>
          </div>
        </div>
      </div>
    </transition>
  </aside>
</template>
<script>
import reProgress from '@/components/elements/core/reProgress/reProgress';
import CollapsableProgressBars from '@/components/elements/filters/CollapsableProgressBars';

export default {
  name: 'sidebar',
  data: () => ({
    loading: false,
  }),
  // TODO clean and typify
  props: {
    metrics: {
      type: Object,
    },
    loadingQ: {
      type: Boolean,
      default: false,
    },
  },
  computed: {
    metricsOverall() {
      const metricsOveral = [];
      Object.entries(this.metrics || {}).forEach((metricEntry) => {
        const metricKey = metricEntry[0];
        const values = Object.values(metricEntry[1]).flatMap(value => Object.values(value).map(parseFloat));
        metricsOveral.push({
          key: metricKey,
          value: (
            values.reduce((sum, e) => e + sum, 0) / values.length
          ).toFixed(2),
        });
      });
      return metricsOveral;
    },
  },
  components: {
    reProgress,
    CollapsableProgressBars,
  },
};
</script>
<style lang="scss">
@import "@/assets/scss/apps/classifier/sidebar.scss";
</style>
