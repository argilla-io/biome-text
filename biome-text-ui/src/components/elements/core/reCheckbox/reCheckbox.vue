<template>
  <div class="re-checkbox" :class="[classes]">
    <label
      :for="id || name"
      class="checkbox-label"
      v-if="$slots.default"
      @click.prevent="toggleCheck"
    >
      <slot></slot>
    </label>
    <div class="checkbox-container" @click.stop="toggleCheck" tabindex="0">
      <input
        type="checkbox"
        :name="name"
        :id="id"
        :disabled="disabled"
        :value="value"
        :checked="checked"
        tabindex="-1"
      >
    </div>
  </div>
</template>

<script>
export default {
  name: 're-checkbox',
  props: {
    name: String,
    value: [String, Boolean, Number, Array],
    id: String,
    disabled: Boolean,
  },
  data() {
    return {
      checked: this.value || false,
    };
  },
  computed: {
    classes() {
      return {
        checked: this.checked,
        disabled: this.disabled,
      };
    },
  },
  watch: {
    value() {
      this.checked = !!this.value;
    },
  },
  methods: {
    toggleCheck($event) {
      if (!this.disabled) {
        this.checked = !this.checked;
        this.$emit('change', this.checked, $event);
        this.$emit('input', this.checked, $event);
      }
    },
  },
};
</script>
<style lang="scss" scoped>
@import "@/assets/scss/components/checkbox.scss";
</style>
