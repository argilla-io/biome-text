<template>
  <div class="spinner" :class="[classes]" :style="styles">
    <svg class="spinner-draw" viewBox="25 25 50 50">
      <defs>
        <linearGradient id="color">
          <stop offset="0%" stop-color="#E7FFEA"></stop>
          <stop offset="40%" stop-color="#A789C3"></stop>
          <stop offset="60%" stop-color="#8500FF"></stop>
          <stop offset="100%" stop-color="#8500FF"></stop>
        </linearGradient>
      </defs>
      <circle
        class="spinner-path"
        cx="50"
        cy="50"
        r="20"
        stroke="url(#color)"
        :stroke-width="mdStroke"
      ></circle>
    </svg>
  </div>
</template>

<script>
export default {
  name: 're-spinner',
  props: {
    mdSize: {
      type: Number,
      default: 50,
    },
    mdStroke: {
      type: Number,
      default: 3,
    },
    mdIndeterminate: Boolean,
    mdProgress: {
      type: Number,
      default: 0,
    },
  },
  computed: {
    classes() {
      return {
        're-indeterminate': this.mdIndeterminate,
      };
    },
    styles() {
      const newSize = `${this.mdSize}px`;

      return {
        width: newSize,
        height: newSize,
      };
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/spinner.scss";
</style>
