<template>
  <div
    class="dropdown"
    v-bind:class="visible ? 'dropdown--open' : ''"
    ref="dropdownMenu"
    v-click-outside="onClickOutside"
  >
    <div class="dropdown__header" @click="onClick">
      <slot name="dropdown-header"></slot>
      <span
        v-bind:class="visible ? 'dropdown__check--open' : ''"
        class="dropdown__check"
        aria-hidden="true"
      ></span>
    </div>
    <div v-show="visible" class="dropdown__content">
      <slot name="dropdown-content"></slot>
    </div>
  </div>
</template>
<script>
export default {
  name: 're-filter-dropdown',
  props: {
    visible: {
      default: false,
      type: Boolean,
    },
  },
  methods: {
    onClickOutside() {
      this.$emit('visibility', false);
    },
    onClick() {
      this.$emit('visibility', !this.visible);
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@recognai/re-commons/src/assets/scss/components/dropdown.scss";
</style>
