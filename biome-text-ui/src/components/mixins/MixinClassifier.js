/* eslint-disable import/no-unresolved */
import Vue from 'vue';
import VueVega from 'vue-vega';
import reEmptyList from '@/components/elements/core/reList/reEmptyList';
import searchbar from '@/components/elements/filters/SearchBar';
import filtersTags from '@/components/elements/filters/FiltersTags';
import filtersList from '@/components/common/FiltersList';
import elasticsearchDataWrapper from '@/components/common/elasticsearch/datawrapper';
import ESClient from '@/api/elasticsearch';

Vue.use(VueVega);

const initFiltersStatus = () => ({});
export default {
  // TODO Normalize names
  data: () => ({
    project: {
      type: String,
    },
    query: '',
    filtersStatus: initFiltersStatus(),
    filtersMeta: initFiltersStatus(),
    paginationSize: 20,
    from: 0,
    sortBy: 'gold',
    sortOrder: 'asc',
    queryFields: undefined,
    totalRecords: 0,
    filtersTagsKey: 0,
    headerHeight: 100,
    datasourceName: undefined,
    model: undefined,
    actionName: undefined,
    showEntityClassifier: false,
    jupyterView: false,
  }),
  props: {
    filename: {
      type: String,
      default: undefined,
    },
    referer: {
      default: Object,
    },
    loadingQ: {
      type: Boolean,
      default: false,
    },
    isInitialLoading: {
      type: Boolean,
      default: false,
    },
    results: {
      type: Object,
      default: () => ({
        total: 0,
        items: [],
        aggregations: {},
      }),
    },
    showHeader: {
      type: Boolean,
      default: true,
    },
    fixedFilters: {
      type: Boolean,
      default: false,
    },
  },
  watch: {
    $route: 'fetchData',
    isInitialLoading() {
      this.getHeaderHeight();
    },
    showHeader() {
      this.getHeaderHeight();
    },
    filters() {
      this.getHeaderHeight();
    },
  },
  methods: {
    onPageChange(page) {
      this.currentPage = page;
      this.emitQueryUpdated();
    },
    onQuery(textQuery) {
      this.query = textQuery;
      this.emitQueryUpdated();
    },
    onClear() {
      this.query = '';
      this.emitQueryUpdated();
    },
    onFilterConfidenceApply(filterId, value) {
      this.$set(this.filtersStatus, filterId, [value]);
      this.emitQueryUpdated();
    },
    onFetchMoreData() {
      this.from += this.paginationSize;
      this.emitQueryUpdated(this.from);
    },
    onApply(filters) {
      this.updateActiveFilters(filters);
      this.emitQueryUpdated();
    },
    onMetaFilterApply(metaFilter) {
      Object.keys(this.filtersStatus)
        .filter(r => (Object.keys(metaFilter).indexOf(r) === -1))
        .filter(f => this.getMetadataKeys.includes(f))
        .forEach((cleanF) => {
          delete this.filtersStatus[cleanF];
        });
      metaFilter.forEach((mf) => {
        const filterKey = mf.key;
        const filterValue = mf.value;
        const currentFilterValue = this.filtersStatus[filterKey] || [];
        if (this.filtersStatus[filterKey] !== currentFilterValue) {
          currentFilterValue.push(filterValue);
        }
        this.$set(this.filtersStatus, filterKey, currentFilterValue);
        this.filtersMeta[filterKey] = {
          id: filterKey,
          values: currentFilterValue,
        };
      });
      this.emitQueryUpdated();
    },
    isActiveFilter(id) {
      return this.filtersStatus[id] && this.filtersStatus[id].length > 0;
    },
    onClearFilter(key, value) {
      this.filtersStatus[key] = this.filtersStatus[key].filter(v => v !== value);
      if (this.filtersStatus[key].length === 0) {
        delete this.filtersStatus[key];
      }
      this.emitQueryUpdated();
      this.forceRerender();
    },
    onClearAll() {
      this.query = '';
      this.filtersStatus = initFiltersStatus();
      this.emitQueryUpdated();
    },
    onSortResults(sortBy, sortOrder) {
      this.sortBy = sortBy;
      this.sortOrder = sortOrder;
      this.emitQueryUpdated();
    },
    updateActiveFilters(selectedFilters) {
      Object.values(selectedFilters).forEach((selectedFilter) => {
        (selectedFilter.values || [])
          .filter(filterValue => filterValue.selected)
          .forEach((filterValue) => {
            const filterStatus = this.filtersStatus[selectedFilter.id] || [];
            filterStatus.push(filterValue.id);
            this.$set(this.filtersStatus, selectedFilter.id, [
              ...new Set(filterStatus),
            ]);
          });
      });
    },
    fetchData() {
      // load filters status
      const routeParams = this.$route.query;
      if (routeParams) {
        Object.keys(routeParams).forEach((filter) => {
          if (routeParams[filter]) {
            switch (filter) {
              case 'search': {
                this.query = routeParams.search;
                break;
              }
              case 'showEntityClassifier': {
                const showEntityClassifier = JSON.parse(routeParams.showEntityClassifier);
                this.showEntityClassifier = showEntityClassifier;
                break;
              }
              case 'jupyterView': {
                const jupyterView = JSON.parse(routeParams.jupyterView);
                this.jupyterView = jupyterView;
                break;
              }
              case 'showlabelled': {
                const showLabelledVal = JSON.parse(routeParams.showlabelled);
                this.viewAllMode = showLabelledVal;
                this.showLabelledAs = showLabelledVal;
                break;
              }
              case 'confidence': {
                const params = `[${routeParams[filter]}]`;
                const paramsArray = JSON.parse(params);
                this.filtersStatus[filter] = [[].concat(paramsArray)];
                break;
              }
              case 'sortby': {
                this.sortBy = routeParams.sortby;
                break;
              }
              case 'sort': {
                this.sortOrder = routeParams.sort;
                break;
              }
              default: {
                this.filtersStatus[filter] = [].concat(routeParams[filter]);
              }
            }
          }
          this.forceRerender();
        });
      }
    },
    routeConfig() {
      // filters to query params
      const routeConfiguration = {
        query: {},
      };
      Object.keys(this.filtersStatus).forEach((filter) => {
        routeConfiguration.query[filter] = this.filtersStatus[filter];
      });
      if (this.query !== '') {
        routeConfiguration.query.search = this.query;
      }
      if (this.viewAllMode === true) {
        routeConfiguration.query.showlabelled = this.viewAllMode;
      }
      if (this.sortBy) {
        routeConfiguration.query.sortby = this.sortBy;
      }
      if (this.sortOrder) {
        routeConfiguration.query.sort = this.sortOrder;
      }
      if (this.showEntityClassifier === true) {
        routeConfiguration.query.showEntityClassifier = this.showEntityClassifier;
      }
      if (this.jupyterView === true) {
        routeConfiguration.query.jupyterView = this.jupyterView;
      }
      this.$router.push(routeConfiguration);
    },
    forceRerender() {
      this.filtersTagsKey += 1;
    },
    getHeaderHeight() {
      this.headerHeight = this.$refs.header.clientHeight;
    },
  },
  computed: {
    breadcrumbs() {
      return [
        { link: this.$route.fullPath, name: this.actionName },
      ];
    },
    confidenceFilterId() {
      return (this.filters.confidence || {}).id;
    },
    filters() {
      return elasticsearchDataWrapper.filtersFromAggregations(
        this.results.aggregations,
      );
    },
    metrics() {
      return elasticsearchDataWrapper.metricsFromAggregation(
        this.results.aggregations,
      );
    },
    getMetadata() {
      const metadata = {};
      if (this.docRecords) {
        this.getMetadataKeys.forEach((key) => {
          let unique = this.docRecords.filter(record => record.metadata[key]).map(record => record.metadata[key]);
          unique = [...new Set(unique)];
          metadata[key] = {
            id: key,
            name: key,
            values: unique.map(uniqueRecord => ({
              id: uniqueRecord,
              name: uniqueRecord,
            })),
          };
        });
      }
      return metadata;
    },
    getMetadataKeys() {
      const firstElement = this.docRecords[0];
      if (firstElement === undefined) {
        return [];
      }
      return Object.keys(firstElement.metadata || []);
    },
    getMetrics() {
      return this.metrics;
    },
    docRecords() {
      return (this.results.items || [])
        // TODO read input configuration
        .map(d => elasticsearchDataWrapper.mapESDocument2Record(d, this.readableFields, this.outputField))
        .slice();
    },
    getDatasourceName() {
      return ESClient.fetchPredictions().then((predictions) => {
        const currentPrediction = predictions.find(p => p.id === this.filename);
        this.datasourceName = currentPrediction.dataSource;
        this.model = currentPrediction.model;
        if (currentPrediction.kind === 'explore') {
          this.actionName = `Explore ${this.$moment(currentPrediction.createdAt).format('MMM Do YYYY H:mm')}`;
        } else {
          this.actionName = currentPrediction.exploreName;
        }
      });
    },
  },
  mounted() {
    const { prediction } = this.$route.params;
    new ESClient(prediction).getTotalRecords().then((r) => {
      this.totalRecords = r;
    });

    ESClient.fetchPrediction(prediction).then((p) => {
      this.queryFields = p.searchableFields;
      this.readableFields = p.readableFields;
      this.outputField = p.output;
      this.emitQueryUpdated();
    });
  },
  components: {
    filtersList,
    filtersTags,
    searchbar,
    reEmptyList,
  },
};
