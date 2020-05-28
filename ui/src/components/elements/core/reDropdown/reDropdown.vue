<template>
  <div
    class="dropdown"
    v-bind:class="visible ? 'dropdown--open' : ''"
    ref="dropdownMenu"
    v-click-outside="onClickOutside"
  >
    <div class="dropdown__header" @click="openDropdown">
      <slot name="dropdown-header"></slot>
      <span
        v-bind:class="visible ? 'dropdown__check--open' : ''"
        class="dropdown__check"
        aria-hidden="true"
      >
        <svgicon :name="visible ? 'drop-up' : 'drop-down'" width="12" height="auto"></svgicon>
      </span>
    </div>
    <div v-show="visible" class="dropdown__content">
      <transition name="fade" appear>
        <slot name="dropdown-content"></slot>
      </transition>
    </div>
  </div>
</template>
<script>
import '@/assets/iconsfont/drop-down';
import '@/assets/iconsfont/drop-up';

export default {
  name: 're-dropdown',
  data() {
    return {
      visible: false,
    };
  },
  methods: {
    openDropdown() {
      this.visible = !this.visible;
    },
    onClickOutside() {
      this.visible = false;
    },
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/dropdown.scss";
</style>
