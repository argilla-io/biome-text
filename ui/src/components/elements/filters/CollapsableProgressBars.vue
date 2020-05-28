<template>
  <div>
    <div class="collapsable__header">
      <p>{{entriesKey}} %</p>
      <span v-if="!hiddenEntries" @click="sort('value')"
        :class="['button', currentSortDir, {'active' : currentSort === 'value'}]"
      >
       {{currentSortDir === 'asc' ? '0 - 100' : '100 - 0'}}
      </span>
    </div>
    <div class="collapsable__body">
      <span
        v-for="(entry, i) in sortedEntries.slice(0, numberOfEntries)"
        :class="entriesKey"
        :key="i"
      >
        <span
          @mouseleave="showTooltipOnHover = false"
          @mouseenter="showTooltip(entry.name, $event)"
          :data-title="entry.name"
        >
          <label>{{entry.name}}</label>
          <span class="progress">{{entry.value}}</span>
          <re-progress
            re-mode="determinate"
            :class="'color_'+entry.id"
            :progress="parseInt(entry.value)"
          ></re-progress>
        </span>
      </span>
    </div>
    <span :class="{'tooltip--visible': showTooltipOnHover}" class="tooltip" ref="tooltip"></span>
          <re-button
        class="button-icon"
        @click="toggleShowCollapse()"
        v-if="entries.length > barsNumber"
      > {{numberOfEntries === undefined ? 'See less' : `See more (${this.hiddenEntries})`}}
        <svgicon
          :name="numberOfEntries === undefined ? 'drop-up' : 'drop-down'"
          width="12"
          height="auto"
        ></svgicon>
      </re-button>
  </div>
</template>

<script>
import reProgress from '@/components/elements/core/reProgress/reProgress';
import reButton from '@/components/elements/core/reButton/reButton';

export default {
  name: 'collapsable-progress-bars',
  props: ['entries', 'entriesKey', 'barsNumber'],
  data: () => ({
    numberOfEntries: '',
    currentSortDir: 'desc',
    currentSort: 'value',
    showTooltipOnHover: false,
  }),
  mounted() {
    this.numberOfEntries = this.barsNumber;
  },
  filters: {
    truncate(string, value) {
      if (string.length > value) {
        return `${string.substring(0, value)}...`;
      }
      return string;
    },
  },
  methods: {
    toggleShowCollapse() {
      if (this.numberOfEntries === this.barsNumber) {
        this.numberOfEntries = undefined;
      } else {
        this.numberOfEntries = this.barsNumber;
      }
    },
    sort(s) {
      if (s === this.currentSort) {
        this.currentSortDir = this.currentSortDir === 'asc' ? 'desc' : 'asc';
      }
      this.currentSort = s;
    },
    showTooltip(data, e) {
      const { tooltip } = this.$refs;
      const el = e.currentTarget;
      if (e.currentTarget && data.length >= 20) {
        tooltip.innerHTML = data;
        this.showTooltipOnHover = true;
        const offset = el.getBoundingClientRect().top - el.offsetParent.getBoundingClientRect().top;
        tooltip.style.top = `${offset - 35}px`;
      } else {
        this.showTooltipOnHover = false;
      }
    },
  },
  computed: {
    hiddenEntries() {
      return this.entryMetrics.length - this.numberOfEntries;
    },
    entryMetrics() {
      const entries = [];
      Object.entries(this.entries || {}).forEach((metricEntry) => {
        const metricKey = Object.entries(metricEntry[1]).flatMap(value => Object.values(value));
        entries.push({
          name: metricKey[0],
          value: parseFloat(metricKey[1]),
          id: metricEntry[0],
        });
      });
      return entries.filter(e => e.value !== 0);
    },
    sortedEntries() {
      const entries = this.entryMetrics;
      return entries.sort((a, b) => {
        let modifier = 1;
        if (this.currentSortDir === 'desc') modifier = -1;
        if (a[this.currentSort] < b[this.currentSort]) return -1 * modifier;
        if (a[this.currentSort] > b[this.currentSort]) return 1 * modifier;

        return 0;
      });
    },
    sortableMetrics() {
      return (
        this.entries.length > this.barsNumber
        && this.numberOfEntries === undefined
      );
    },
  },
  components: {
    reProgress,
    reButton,
  },
};
</script>

<style lang="scss" scoped>
// @import "@/assets/scss/apps/classifier/sidebar.scss";
@import "@/assets/scss/components/tooltip.scss";
.collapsable__header {
  display: flex;
  position: absolute;
  width: 100%;
  z-index: 1;
  .button {
    float: right;
    color: palette(orange);
    margin-right: 0;
    margin-left: auto;
    cursor: pointer;
    &.active.desc {
        &:after {
            transform: rotate(180deg);
            transform-origin: 50% 100%;
        }
    }
    &:after {
        content: '▾';
        display: inline-block;
        height: 10px;
        width: 16px;
        box-sizing: inherit;
        text-align: center;
        margin-left: 1em;
    }
  }
}
.collapsable__body {
  padding-top: 2em;
}
</style>
