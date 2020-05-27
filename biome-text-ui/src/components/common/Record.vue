<template>
  <div>
    <record-explain v-if="explain" :query="query" :predictedOk="predictedOk" :explain="explain"></record-explain>
    <span v-else v-for="(recordValue, index) in filteredRecord" :key="index" class="record">
      <record-text v-if="isStringOrNumber(recordValue)" :query="query" :showEntityClassifier="showEntityClassifier" :record="record" :index="index" :recordValue="recordValue"></record-text>
      <record-list v-else-if="isList(recordValue)" :query="query" :showEntityClassifier="showEntityClassifier" :record="record" :index="index" :recordValue="recordValue"></record-list>
      <span v-else class="record__item" v-for="(value, index) in recordValue" :key="index">
        <span class="record__key">{{index}}</span>:ddd
        <span v-html="highlightSearch(value, this.query)"></span>
      </span>
    </span>
  </div>
</template>
<script>
import recordList from '@/components/common/RecordList';
import recordText from '@/components/common/RecordText';
import recordExplain from '@/components/common/RecordExplain';

export default {
  name: 'record',
  props: {
    record: {
      type: Object,
    },
    showEntityClassifier: {
      type: Boolean,
      default: false,
    },
    predictedOk: {
      type: Boolean,
      default: true,
    },
    explain: {
      type: Object,
    },
    query: {
      type: String,
      default: undefined,
    },
  },
  methods: {
    isStringOrNumber(recordValue) {
      if (typeof (recordValue) === 'string' || typeof (recordValue) === 'number') {
        return true;
      }
      return false;
    },
    isList(recordValue) {
      if (Array.isArray(recordValue)) {
        return true;
      }
      return false;
    },
    highlightSearch(option, searchText) {
      if (!searchText) {
        return option;
      }
      return option.toString().replace(new RegExp(searchText), match => `<span class="highlight-text">${match}</span>`);
    },
  },
  computed: {
    filteredRecord() {
      const filter = excludeEntries => Object.keys(this.record)
        .filter(key => !excludeEntries.includes(key))
        .reduce((obj, key) => {
          const o = obj;
          o[key] = this.record[key];
          return o;
        },
        {});
      if (this.record['tokens.context_type'] === 'tabular') {
        const hidden = ['tokens.context_type', 'tokens.position', 'tokens.inputs.value'];
        return filter(hidden);
      } if (this.record['tokens.context_type'] === 'textual') {
        const hidden = ['tokens.context_type', 'tokens.inputs.value', 'tokens.inputs.label', 'tokens.inputs.span_end', 'tokens.inputs.span_start'];
        return filter(hidden);
      }
      return this.record;
    },
  },
  components: {
    recordList,
    recordText,
    recordExplain,
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/record.scss";
</style>
