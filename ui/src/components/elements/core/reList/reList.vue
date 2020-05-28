<template>
  <ul :class="[sorted ? 'list--sorted' : '' , 'list']">
    <slot></slot>
    <transition-group name="list">
      <slot name="list"></slot>
    </transition-group>
  </ul>
</template>
<script>
export default {
  name: 're-list',
  props: ['sorted'],
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/list.scss";
</style>
