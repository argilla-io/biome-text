<template>
  <div id="ExploreClassifier" :class="{'theme-jupyter' : jupyterView}">
    <!-- filters -->
    <section
      ref="header"
      :class="['header--explore', 'header', !showHeader ? 'header--hidden' : '']"
    >
      <re-topbar-brand v-if="!jupyterView">
        <re-breadcrumbs-area  :breadcrumbs="breadcrumbs"></re-breadcrumbs-area>
      </re-topbar-brand>
      <div class="filters__area">
        <div class="filters__content">
          <div class="container">
            <div class="filters__row">
              <div class="filters__title">
                <!-- <h1 class="filters__title__action">
                  {{actionName}}
                </h1>-->
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
                  <!-- <span v-if="model !== 'none'" :title="model">
                    with
                    <span class="filters__title__model">{{model}}</span>
                  </span> -->
                  <span class="filters__title__records">({{results.total}} Records)</span>
                </h2>
              </div>
              <searchbar class="filters__searchbar" :query="query" @submit="onQuery"></searchbar>
              <!-- <div class="filters__records-number" v-if="jupyterView">{{results.total}} records</div>
              <biome-logo class="biome-logo" v-if="jupyterView" :height="20" :width="70"/> -->
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
import reTopbarBrand from '@/components/elements/core/reTopbar/reTopbarBrand';
import reBreadcrumbsArea from '@/components/elements/core/reBreadcrumbsArea/reBreadcrumbsArea';
// import biomeLogo from '@/components/elements/core/logos/biomeLogo';
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
    if (this.jupyterView) {
      const iframe = window.self !== window.top ? 'iframe' : 'browser';
      document.querySelector('body').classList.add('theme-jupyter', iframe);
    }
  },
  components: {
    sidebar,
    searchResults,
    reTopbarBrand,
    reBreadcrumbsArea,
    // biomeLogo,
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
body.theme-jupyter {
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
}
</style>
