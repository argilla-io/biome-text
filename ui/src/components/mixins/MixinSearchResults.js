/* eslint-disable import/no-unresolved */

import Vue from 'vue';
import VueWaypoint from 'vue-waypoint';
import reDropdown from '@/components/elements/core/reDropdown/reDropdown';
import reButton from '@/components/elements/core/reButton/reButton';
import reNumeric from '@/components/elements/core/format/reNumeric/reNumeric';
import reProgress from '@/components/elements/core/reProgress/reProgress';
import reModal from '@/components/elements/core/reModal/reModal';
import metadata from '@/components/common/Metadata';
import record from '@/components/common/Record';
import loadingSkeleton from '@/components/common/LoadingSkeleton';

Vue.use(VueWaypoint);

export default {
  name: 'search-results',
  data: () => ({
    scrolling: false,
    scrollerPosition: 0,
    metaFiltrable: true,
    scrollIndex: 0,
    showMoreDataButton: false,
    showMetadata: undefined,
    checkPageNumber: {
      root: null,
      rootMargin: '20px',
      threshold: [0],
    },
  }),
  props: {
    total: {
      type: Number,
      default: undefined,
    },
    records: {
      type: Array,
    },
    filtersStatus: {
      type: Object,
      default: () => { },
    },
    loadingQ: {
      type: Boolean,
      default: false,
    },
    showEntityClassifier: {
      type: Boolean,
      default: false,
    },
    query: {
      type: String,
      default: undefined,
    },
  },
  methods: {
    moreData() {
      this.$emit('fetchMoreData');
    },
    onUpdate(start, end) {
      this.scrollIndex = start;
      if (end === this.filteredRecords.length) {
        this.showMoreDataButton = true;
      } else {
        this.showMoreDataButton = false;
      }
    },
    handleScroll() {
      if (window.scrollY > 100) {
        this.scrolling = true;
        const scrollTop = window.pageYOffset
          || document.documentElement.scrollTop
          || document.body.scrollTop
          || 0;
        const { body } = document;
        const html = document.documentElement;

        const height = Math.max(
          body.scrollHeight,
          body.offsetHeight,
          html.clientHeight,
          html.scrollHeight,
          html.offsetHeight,
        );
        const w = window;
        const d = document;
        const e = d.documentElement;
        const g = d.getElementsByTagName('body')[0];
        const wheight = w.innerHeight || e.clientHeight || g.clientHeight;
        const scrollPercent = 100 * scrollTop / (height - wheight);
        const marginTop = (wheight - 20) / 100 * scrollPercent;
        this.scrollerPosition = marginTop;
      } else {
        this.scrolling = false;
      }
    },
    onMetaFilterApply(metaFilter) {
      this.$emit('metafilterapply', metaFilter);
      this.showMetadata = false;
    },
    decorateConfidence(confidence) {
      return confidence * 100;
    },
    closeModal() {
      this.showMetadata = false;
    },
    onShowExplain(id) {
      if (this.showExplain === id) {
        this.showExplain = undefined;
      } else {
        this.showExplain = id;
      }
    },
    onWaypoint({ el, going }) {
      const pages = document.querySelectorAll('.list__item');
      if (going === this.$waypointMap.GOING_IN) {
        pages.forEach((page) => {
          page.classList.remove('--visible');
        });
        el.classList.add('--visible');
      }
    },
  },
  computed: {
    shownRecordsProgress() {
      const progress = 100 * this.records.length / this.total;
      return progress;
    },
    shownRecords() {
      return this.records;
    },
  },
  created() {
    window.addEventListener('scroll', this.handleScroll);
  },
  destroyed() {
    window.removeEventListener('scroll', this.handleScroll);
  },
  components: {
    metadata,
    reDropdown,
    reNumeric,
    reButton,
    reProgress,
    loadingSkeleton,
    reModal,
    record,
  },
};
