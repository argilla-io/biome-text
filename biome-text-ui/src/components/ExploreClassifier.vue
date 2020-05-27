<template>
  <div id="ExploreClassifier" :class="{'theme-jupyter' : jupyterView}">
    <!-- filters -->
    <section
      ref="header"
      :class="['header--explore', 'header', !showHeader ? 'header--hidden' : '']"
    >
      <div class="filters__area">
        <div class="filters__content">
          <div class="container">
            <div class="filters__row">
              <div class="filters__title">
                <h2
                  :getDatasourceName="getDatasourceName"
                  v-if="results"
                  class="filters__title__info"
                >
                  <svgicon name="datasource" width="20" height="14"></svgicon>
                  <span
                    class="filters__title__datasource"
                    :title="datasourceName"
                  >{{datasourceName}}</span>
                  <span class="filters__title__records">({{results.total}} Records)</span>
                </h2>
              </div>
              <searchbar class="filters__searchbar" :query="query" @submit="onQuery"></searchbar>
            </div>
            <!-- filters list -->
            <filters-list
              :filters="filters"
              :filtersStatus="filtersStatus"
              :getMetadata="getMetadata"
              @filterConfidenceApply="onFilterConfidenceApply"
              @apply="onApply"
              @sort="onSortResults"
            ></filters-list>
          </div>
        </div>
        <filters-tags
          :key="filtersTagsKey"
          :query="query"
          :filtersStatus="filtersStatus"
          @clearFilter="onClearFilter"
          @clearAll="onClearAll"
          @clearQuery="onClear"
        ></filters-tags>
      </div>
    </section>
    <!-- results main -->
    <div class="container">
      <transition name="fade" appear>
        <div
          class="grid"
          :style="{paddingTop: !showHeader ? this.headerHeight + 'px' : ''}"
          :class="[{'grid--fixed-mode' : fixedFilters}]"
        >
          <re-empty-list v-if="results.total === 0" empty-title="0 results found"></re-empty-list>
          <search-results
            v-else
            :showEntityClassifier="showEntityClassifier"
            :loadingQ="loadingQ"
            :records="docRecords"
            :jupyterView="jupyterView"
            :allowInfiniteScroll="fixedFilters"
            :filtersStatus="filtersStatus"
            :total="getTotalQuery"
            :query="query"
            @metafilterapply="onMetaFilterApply"
            @fetchMoreData="onFetchMoreData"
          ></search-results>
          <!-- metrics -->
          <sidebar
            :class="[{'sidebar--fixed' : fixedFilters}, {'sidebar--fixed-top' : !showHeader && fixedFilters}]"
            :loadingQ="loadingQ"
            v-if="results.total !== 0"
            :metrics="getMetrics"
            :total-items="getTotal"
          ></sidebar>
        </div>
      </transition>
    </div>
  </div>
</template>

<script>
import searchResults from './explore/SearchResults';
import sidebar from './explore/SideBar';
import elasticsearch from './common/elasticsearch/queries';
import MixinClassifier from './mixins/MixinClassifier';

export default {
  mixins: [MixinClassifier],
  name: 'explore-results-classifier',
  created() {
    this.fetchData();
  },
  methods: {
    emitQueryUpdated(from) {
      if (!from) {
        this.from = 0;
      }
      const query = this.buildQuery(from);
      this.$emit('queryChanged', query);
      this.routeConfig();
    },
    buildQuery(from) {
      const query = {
        keyword: this.query,
        queryFields: this.queryFields,
        filtersStatus: this.filtersStatus,
        esOptions: {
          from: from || 0,
          size: this.paginationSize,
        },
        sortOptions: {
          sortBy: this.sortBy,
          sortOrder: this.sortOrder,
        },
        hasGold: true,
      };
      return elasticsearch.toESQuery({ ...query, showAll: true });
    },
  },
  computed: {
    getTotal() {
      return this.totalRecords;
    },
    getTotalQuery() {
      return this.results.total;
    },
  },
  mounted() {
    const iframe = window.self !== window.top ? 'iframe' : 'browser';
    document.querySelector('body').classList.add('theme-jupyter', iframe);
  },
  components: {
    sidebar,
    searchResults,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/filters.scss";
@import "@/assets/scss/apps/classifier/grid.scss";
@import "@/assets/scss/components/breadcrumbs.scss";
@import "@/assets/scss/apps/jupyter/theme-jupyter.scss";
</style>
<style lang="scss">
body {
  &:after {
    content: '';
    border: 3px solid palette(orange);
    position: fixed;
    width: 100vw;
    height: 100vh;
    z-index: 9999;
    top: 0;
    pointer-events: none;
  }
  .filters {
    &__searchbar {
      .searchbar .re-input {
        @include font-size(12px);
        color: palette(grey, dark);
        @include input-placeholder {
          color: palette(grey, dark);
        }
      }
    }
  }
  .filters__tags  {
    border-bottom: 0 !important
  }
  .filter__sort, .filter__show-more {
    min-width: 100px;
  }
  .show-more-data {
    background: none !important
  }
  &.iframe {
    .main, .container {
      max-width: none !important
    }
  }
  .sidebar__wrapper {
    top: 0 !important
  }
}
</style>
