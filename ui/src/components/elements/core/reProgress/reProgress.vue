<template>
  <transition name="re-progress" appear>
    <div class="re-progress">
      <div class="re-progress-track" :style="styles">
        <div v-if="multiple" class="re-progress-track--secondary" :style="stylesSecondary"></div>
      </div>
    </div>
  </transition>
</template>

<script>
export default {
  name: 're-progress',
  props: {
    indeterminate: Boolean,
    progress: {
      type: Number,
      default: 0,
    },
    progressSecondary: {
      type: Number,
      default: 0,
    },
    multiple: {
      default: false,
      type: Boolean,
    },
  },
  computed: {
    classes() {
      return {
        're-indeterminate': this.indeterminate,
      };
    },
    styles() {
      if (!this.indeterminate) {
        return {
          width: `${this.progress}%`,
        };
      }
      return undefined;
    },
    stylesSecondary() {
      if (!this.indeterminate) {
        return {
          width: `${this.progressSecondary}%`,
        };
      }
      return undefined;
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/progress.scss";
</style>
