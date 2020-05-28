<template>
  <transition mode="in-out" :duration="{ enter: 0, leave: 500 }" name="skeleton" appear>
  <div class="loading-skeleton" v-if="condition">
    <span class="loading-skeleton__line"></span>
  </div>
  </transition>
</template>

<script>
import '@/assets/iconsfont/drop-down';

export default {
  name: 'loading-skeleton',
  props: {
    itemsNumber: {
      type: Number,
      default: 10,
    },
    buttonsNumber: {
      type: Number,
      default: 4,
    },
    condition: {
      type: Boolean,
      default: false,
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/loading-skeleton.scss";
</style>
