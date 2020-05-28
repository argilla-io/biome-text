<template>
  <div class="breadcrumbs__area">
    <div class="breadcrumbs__area__content">
      <re-breadcrumbs :breadcrumbs="breadcrumbs"></re-breadcrumbs>
      <div class="breadcrumbs__area--right">
        <slot name="external-links"></slot>
      </div>
    </div>
  </div>
</template>

<script>
import reBreadcrumbs from '@/components/elements/core/reBreadcrumbsArea/reBreadcrumbs';

export default {
  name: 're-breadcrumbs-area',
  props: {
    breadcrumbs: {
      type: Array,
    },
  },
  components: {
    reBreadcrumbs,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/apps/classifier/grid.scss";
@import "@/assets/scss/components/breadcrumbs.scss";
</style>
