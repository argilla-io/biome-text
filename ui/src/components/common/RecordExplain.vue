<template>
  <div>
    <span class="record" v-for="input in Object.keys(explain)" :key="input.index">
      <span class="record__key--explain">{{input}}</span>
      <explain :query="query" :predictedOk="predictedOk" :interpret="explain[input]"></explain>
    </span>
  </div>
</template>
<script>
import explain from '@/components/common/Explain';

export default {
  data: () => ({
    // allowScroll: false,
  }),
  name: 'record-explain',
  props: {
    showEntityClassifier: {
      type: Boolean,
      default: false,
    },
    explain: {
      type: Object,
    },
    query: {
      type: String,
      default: undefined,
    },
    predictedOk: {
      type: Boolean,
      default: true,
    },
  },
  components: {
    explain,
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/record.scss";
</style>
