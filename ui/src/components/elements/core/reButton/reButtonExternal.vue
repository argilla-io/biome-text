<template>
  <div class="external-link">
    <router-link target='_blank' :to="link">{{title}}</router-link>
    <svgicon :name="icon" width="12" height="12"></svgicon>
  </div>
</template>

<script>
import '@/assets/iconsfont/external-link';

export default {
  name: 're-button-external',
  props: {
    link: {
      type: Object,
    },
    icon: {
      type: String,
      default: 'external-link',
    },
    title: {
      type: String,
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/buttons.scss";
</style>
