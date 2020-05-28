<template>
  <transition name="fade">
    <div class="panel" :class="{hidden: panelCloseButton}">
      <p class="panel__text">
        <slot></slot>
      </p>
      <re-button v-if="showCloseButton" class="panel-close" @click="onButtonClicked = true">
        <svgicon name="cross" width="12" height="auto"></svgicon>
      </re-button>
    </div>
  </transition>
</template>

<script>
import '@/assets/iconsfont/cross';
import reButton from '@/components/elements/core/reButton/reButton';

export default {
  name: 're-panel',
  data() {
    return {
      panelCloseButton: false,
    };
  },
  props: ['showCloseButton'],
  methods: {
    onButtonClicked() {
      this.panelCloseButton = true;
    },
  },
  components: {
    reButton,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/panel.scss";
</style>
