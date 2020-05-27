<template>
  <ul class="breadcrumbs">
    <li>
      <router-link v-for="breadcrumb in breadcrumbs" :key="breadcrumb.name" class="breadcrumbs__item" :to="breadcrumb.link">{{breadcrumb.name}}</router-link>
    </li>
  </ul>
</template>

<script>
export default {
  name: 're-breadcrumbs',
  props: {
    breadcrumbs: {
      type: Array,
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/breadcrumbs.scss";
</style>
