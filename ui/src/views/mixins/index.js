/* eslint-disable import/no-unresolved */
import reLoading from '@/components/elements/core/reLoading/reLoading';
import ESClient from '@/api/elasticsearch';

export default {
  data: () => ({
    esResults: {
      aggregations: {},
      hits: {
        hits: [],
        total: 0,
      },
    },
    showTopbar: true,
    lastScrollPosition: 0,
    hideTopbarOn: 1,
    lastScrollDistance: 235,
    isInitialLoading: true,
    loadingQ: false,
    showHeader: true,
    scrollLarge: false,
    referer: {
      name: 'project',
    },
    timer: null,
  }),
  events: {},
  props: {
    prediction: String,
    inputs: {
      type: Array,
      default: () => [],
    },
  },
  methods: {
    onScroll() {
      // this.checkScrollSpeed();
      const currentScrollPosition = window.pageYOffset || document.documentElement.scrollTop;
      this.showTopbar = currentScrollPosition < this.hideTopbarOn;
      if (currentScrollPosition > this.lastScrollDistance) {
        this.showHeader = false;
        if (this.scrollLarge === true) {
          this.showHeader = true;
        }
      } else {
        this.showHeader = true;
      }
    },
    async onQueryChanged(query) {
      if (query.from === 0) {
        try {
          this.loadingQ = true;
          const results = await this.elasticsearch.search(query);
          this.esResults = results;
        } finally {
          this.loadingQ = false;
        }
      } else if (this.results.total > this.results.items.length) {
        try {
          this.loadingQ = true;
          const results = await this.elasticsearch.search(query);
          results.hits.hits = this.esResults.hits.hits.concat(results.hits.hits);
          this.esResults = results;
        } finally {
          this.loadingQ = false;
        }
      }
    },

    updateFeedbackAggregation({ key, count }) {
      const results = this.esResults;
      const keyFound = results.aggregations.feedbackStatus.buckets.find(e => e.key === key);
      if (keyFound) {
        keyFound.doc_count += count;
      } else {
        results.aggregations.feedbackStatus.buckets.push({
          key,
          doc_count: count,
        });
      }
      this.esResults = results;
    },
  },
  computed: {
    filename() {
      return this.esIndex;
    },
    results() {
      return {
        aggregations: this.esResults.aggregations,
        items: this.esResults.hits.hits,
        total: this.esResults.hits.total,
      };
    },
    esIndex() {
      if (this.prediction) {
        return this.prediction;
      }
      return window.location.pathname.slice(1);
    },
    elasticsearch() {
      return new ESClient(this.esIndex);
    },
  },
  updated() {
    if (this.esResults.timed_out === false) {
      this.isInitialLoading = false;
    }
  },
  mounted() {
    window.addEventListener('scroll', this.onScroll);
  },
  beforeDestroy() {
    window.removeEventListener('scroll', this.onScroll);
  },
  components: {
    reLoading,
  },
};
