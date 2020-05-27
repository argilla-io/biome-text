<template>
  <a
    class="re-button"
    :class="{'loading' : loading, 'centered' : centered}"
    :href="href"
    :loading="loading"
    :disabled="disabled"
    :target="target"
    :rel="newRel"
    @click="$emit('click', $event)"
    v-if="href"
  >
    <re-spinner v-if="loading"></re-spinner>
    <slot></slot>
  </a>

  <button
    class="re-button"
    :class="{'loading' : loading, 'centered' : centered}"
    tabindex="0"
    :loading="loading"
    :type="type"
    :disabled="disabled"
    @click="$emit('click', $event)"
    v-else
  >
    <re-spinner v-if="loading"></re-spinner>
    <slot></slot>
  </button>
</template>

<script>
import reSpinner from '@/components/elements/core/reSpinner/reSpinner';

export default {
  name: 're-button',
  props: {
    href: String,
    target: String,
    rel: String,
    type: {
      type: String,
      default: 'button',
    },
    loading: Boolean,
    disabled: Boolean,
    centered: Boolean,
  },
  computed: {
    newRel() {
      if (this.target === '_blank') {
        return this.rel || 'noopener';
      }

      return this.rel;
    },
  },
  components: {
    reSpinner,
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/buttons.scss";
</style>
