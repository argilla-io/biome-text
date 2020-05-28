<template>
  <div class="topbar">
    <div class="container topbar__container">
      <svgicon v-if="!title"
        class="branding__recognai"
        name="brand-biome"
        width="40"
        height="auto"
        color="#ffffff"
      ></svgicon>
      <router-link :to="{name: 'projects'}" v-else>
        <p class="title">
          {{title}}
        </p>
      </router-link>
      <router-link v-if="referer"
        class="re-button topbar__button"
        :to="referer"
      >
        Back<svgicon name="back" width="14" height="auto"></svgicon>
      </router-link>
      <slot></slot>
    </div>
  </div>
</template>

<script>
import '@/assets/iconsfont/brand-biome';
import '@/assets/iconsfont/back';

export default {
  name: 're-topbar-brand',
  props: {
    title: {
      type: String,
      default: undefined,
    },
    referer: {
      type: Object,
    },
    icon: {
      type: String,
      default: undefined,
    },
  },
  data: () => ({}),
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/topbar.scss";
@import "@/assets/scss/apps/classifier/grid.scss";
.title {
  color: palette(white);
  text-decoration: none;
  .svg-icon {
    margin-right: 0.5em;
  }
}
</style>
