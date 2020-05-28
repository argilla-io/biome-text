<template>
  <div class="filters__tags" :class="{'filters__tags--empty' : !Object.keys(filtersStatus).length}">
    <div v-for="(filterItem, key) in filtersStatus" :key="key">
      <span class="tag" v-for="(value,idx) in filterItem" :key="idx">
        <span>{{tagKey(key)}} = {{decorateTagValue(value)}}</span>
        <i
          aria-hidden="true"
          tabindex="1"
          class="tag-icon"
          @click="$emit('clearFilter', key, value)"
        ></i>
      </span>
    </div>
    <span class="tag" v-if="query">
      <span>Search = {{query}}</span>
      <i aria-hidden="true" tabindex="1" class="tag-icon" @click="$emit('clearQuery', query)"></i>
    </span>

    <span v-if="severalFiltersApplied" class="tag tag--all">
      <span>Clear all</span>
      <i aria-hidden="true" tabindex="1" class="tag-icon" @click="$emit('clearAll')"></i>
    </span>
  </div>
</template>
<script>
export default {
  name: 'filters-tags',
  props: {
    filtersStatus: Object,
    query: String,
  },
  methods: {
    decorateTagValue(value) {
      if (Array.isArray(value)) {
        return value.map(v => `${v}%`).join(' - ');
      }
      return value;
    },
    tagKey(key) {
      if (key === 'gold' || key === 'feedbackStatus') {
        return 'Labelled as';
      }
      if (key === 'predicted') {
        return 'Predicted as';
      }
      if (key === 'confidence') {
        return 'Confidence';
      }
      return key;
    },
  },
  computed: {
    severalFiltersApplied() {
      let sum = Object.values(this.filtersStatus)
        .map(value => value.length)
        .reduce((a, b) => a + b, 0);
      if (this.query) {
        sum += 1;
      }
      return sum > 1;
    },
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/filters-tags.scss";
</style>
