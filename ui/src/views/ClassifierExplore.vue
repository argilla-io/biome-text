<template>
  <div>
    <re-loading v-show="isInitialLoading"></re-loading>
    <div id="exploreResults" class="biome-app" v-show="!isInitialLoading">
      <ExploreClassifier
        :loadingQ="loadingQ"
        :filename="filename"
        :inputs="inputs"
        :showHeader="showHeader"
        :fixedFilters="!showTopbar"
        :results="results"
        :isInitialLoading="isInitialLoading"
        :referer="referer"
        @queryChanged="onQueryChanged"
      ></ExploreClassifier>
      <re-footer></re-footer>
    </div>
  </div>
</template>

<script>
import reFooter from '@/components/elements/core/reFooter/reFooter';
import ExploreClassifier from '@/components/ExploreClassifier';
import mixins from './mixins';

export default {
  name: 'classifier-explore',
  mixins: [mixins],
  data: () => ({
  }),
  components: {
    ExploreClassifier,
    reFooter,
  },
};
</script>

<style lang="scss">
</style>
