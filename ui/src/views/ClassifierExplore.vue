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
    </div>
  </div>
</template>

<script>
import ExploreClassifier from '@/components/ExploreClassifier';
import mixins from './mixins';

export default {
  name: 'classifier-explore',
  mixins: [mixins],
  data: () => ({
  }),
  components: {
    ExploreClassifier,
  },
};
</script>

<style lang="scss">
</style>
