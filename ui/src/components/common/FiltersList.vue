<template>
  <div id="SearchFilterList" class="filters__container grid">
    <div class="filters">
      <div class="filter" v-if="isFilterAvailable(goldFilter)" :class="[predictedFilter.values.length ? '' : 'disabled', filtersStatus.gold ? '--selected' : '']">
        <div>
          <select-filter
            name="Labelled as"
            :filter="goldFilter"
            @filter-changed="onFilterChanged"
            @apply="onApplyOn(goldFilter.id)"
          ></select-filter>
        </div>
      </div>
      <div class="filter" v-if="isFilterAvailable(feedbackStatusFilter)" :class="filtersStatus.feedbackStatus ? '--selected' : ''">
        <div>
          <select-filter
            :class="showLabelledAs ? '' : 'disabled-filter'"
            name="Labelled as"
            :filter="feedbackStatusFilter"
            @filter-changed="onFilterChanged"
            @apply="onApplyOn(feedbackStatusFilter.id)"
          ></select-filter>
        </div>
      </div>
      <div
        class="filter"
        v-if="isFilterAvailable(predictedFilter)"
        :class="[predictedFilter.values.length ? '' : 'disabled', filtersStatus.predicted ? '--selected' : '']"
      >
        <div>
          <select-filter
            name="Predicted as"
            :filter="predictedFilter"
            @filter-changed="onFilterChanged"
            @apply="onApplyOn(predictedFilter.id)"
          ></select-filter>
        </div>
      </div>
      <div class="filter" v-if="isFilterAvailable(confusionMatrixFilter)" :class="[filtersStatus.predicted || filtersStatus.gold ? '--selected' : '']">
        <filter-confusion-matrix :filter="confusionMatrixFilter" @apply="onFilterMatrixApply"></filter-confusion-matrix>
      </div>
      <div class="filter" v-if="isFilterAvailable(confidenceFilter)" :class="[filtersStatus.confidence ? '--selected' : '']">
        <filter-confidence :filter="confidenceFilter" @apply="onFilterConfidenceApply"></filter-confidence>
      </div>
      <transition appear name="fade">
      <div
        class="filter" v-if="(getMetadata && showAllFilters) || (getMetadata && this.filtersNumber < 4)"
        :class="[getMetadata ? '' : 'disabled', areMetadataSelected ? '--selected-meta' : '']"
      >
        <div>
          <select-filter
            name="Metadata"
            :multilevel="true"
            :filtersStatus="filtersStatus"
            :filter="getMetadata"
            @filter-changed="onFilterChanged"
            @apply="onApplyOn(getMetadata)"
          ></select-filter>
        </div>
      </div>
      </transition>
    </div>
    <div class="filters--right">
      <div v-if="this.filtersNumber > 3" :class="['filter__show-more', showAllFilters ? 'active' : '']" @click="showMoreFilters()">{{showAllFilters ? 'Less' : 'More'}} filters</div>
      <div class="filter filter__sort" v-if="sortable">
        <re-sort-list @sort="onSort"></re-sort-list>
      </div>
    </div>
  </div>
</template>

<script>
import SelectFilter from '@/components/elements/filters/SelectFilter';
import FilterConfusionMatrix from '@/components/elements/filters/FilterConfusionMatrix';
import FilterConfidence from '@/components/elements/filters/FilterConfidence';
import reSortList from '@/components/elements/core/reSortList/reSortList';

export default {
  name: 'FilterList',
  props: {
    filters: Object,
    filtersStatus: Object,
    sortable: {
      type: Boolean,
      default: true,
    },
    showLabelledAs: {
      type: Boolean,
      default: true,
    },
    getMetadata: Object,
    feedbackMetrics: Object,
  },
  data: () => ({
    sortBy: 'gold',
    sortByDir: 'desc',
    filtersChanged: {},
    filtersNumber: undefined,
    showAllFilters: true,
  }),
  watch: {
    filters() {
      this.filtersChanged = {};
      this.filtersNumber = Object.keys(this.filters).length;
    },
  },
  computed: {
    goldFilter() {
      return this.filters.gold;
    },
    predictedFilter() {
      return this.filters.predicted;
    },
    feedbackStatusFilter() {
      return this.filters.feedbackStatus;
    },
    confidenceFilter() {
      return this.filters.confidence;
    },
    confusionMatrixFilter() {
      return this.filters.confusionMatrix;
    },
    areMetadataSelected() {
      let areSelected;
      Object.keys(this.filtersStatus).forEach((filter) => {
        if (Object.prototype.hasOwnProperty.call(this.getMetadata, filter)) {
          areSelected = true;
        }
      });
      return areSelected;
    },
  },
  mounted() {
    setTimeout(() => {
      this.$nextTick(() => {
        if (this.filtersNumber <= 4) {
          this.showAllFilters = true;
        } else {
          this.showAllFilters = false;
        }
      });
    }, 0);
  },
  methods: {
    showMoreFilters() {
      if (!this.showAllFilters) {
        this.showAllFilters = true;
      } else {
        this.showAllFilters = false;
      }
    },
    isFilterAvailable(filter) {
      return filter && filter.values;
    },
    onFilterChanged(filterId, filterValue, isActive) {
      let filterList;
      if (this.filters[filterId]) {
        filterList = this.filters;
      } else {
        filterList = this.getMetadata;
      }
      const valueId = filterList[filterId].values.findIndex(
        value => value.id === filterValue,
      );
      const value = filterList[filterId].values[valueId];
      value.selected = isActive;
      const filter = this.filtersChanged[filterId] || { id: filterId, values: [] };
      filter.values[valueId] = value;
      this.filtersChanged[filterId] = filter;
    },
    onFilterMatrixApply(selected) {
      Object.entries(selected).forEach((entry) => {
        const filterId = entry[0];
        const filterValue = entry[1];
        this.onFilterChanged(filterId, filterValue, true);
      });
      this.onApplyOn();
    },
    onFilterConfidenceApply(ranges) {
      this.$emit('filterConfidenceApply', this.filters.confidence.id, ranges);
    },
    onApplyOn() {
      this.$emit('apply', this.filtersChanged);
    },
    onSort(currentSort, currentSortDir) {
      this.$emit('sort', currentSort, currentSortDir);
    },
    filterActive(filter) {
      return filter && filter.length > 0;
    },
  },
  components: {
    FilterConfusionMatrix,
    FilterConfidence,
    SelectFilter,
    reSortList,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/filters.scss";
</style>
