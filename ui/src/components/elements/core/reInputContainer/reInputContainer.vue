<template>
  <div class="re-input-container" :class="[classes]">
    <slot></slot>

    <span class="re-count" v-if="enableCounter">{{ inputLength }} / {{ counterLength }}</span>

    <button
      tabindex="-1"
      type="button"
      class="button-icon"
      @click.prevent="togglePasswordType"
      v-if="reHasPassword"
    >
      <svgicon
        v-bind:class="[showPassword ? 'hide' : 'show']"
        name="password-show"
        width="16"
        height="auto"
        color="#333333"
      ></svgicon>
      <svgicon
        v-bind:class="[showPassword ? 'show' : 'hide']"
        name="password-hide"
        width="16"
        height="auto"
        color="#333333"
      ></svgicon>
    </button>

    <button tabindex="-1" class="button-icon" @click="clearInput" v-if="reClearable && hasValue">
      <re-icon>clear</re-icon>
    </button>
  </div>
</template>

<script>
import isArray from '@/components/core/utils/isArray';
import '@/assets/iconsfont/password-hide';
import '@/assets/iconsfont/password-show';

export default {
  name: 're-input-container',
  props: {
    reInline: Boolean,
    reHasPassword: Boolean,
    reClearable: Boolean,
  },
  data() {
    return {
      value: '',
      input: false,
      inputInstance: null,
      showPassword: false,
      enableCounter: false,
      hasSelect: false,
      hasPlaceholder: false,
      hasFile: false,
      isDisabled: false,
      isRequired: false,
      isFocused: false,
      counterLength: 0,
      inputLength: 0,
    };
  },
  computed: {
    hasValue() {
      if (isArray(this.value)) {
        return this.value.length > 0;
      }

      return Boolean(this.value);
    },
    classes() {
      return {
        're-input-inline': this.reInline,
        're-has-password': this.reHasPassword,
        're-clearable': this.reClearable,
        're-has-select': this.hasSelect,
        're-has-file': this.hasFile,
        're-has-value': this.hasValue,
        're-input-placeholder': this.hasPlaceholder,
        're-input-disabled': this.isDisabled,
        're-input-required': this.isRequired,
        're-input-focused': this.isFocused,
      };
    },
  },
  methods: {
    isInput() {
      return this.input && this.input.tagName.toLowerCase() === 'input';
    },
    togglePasswordType() {
      if (this.isInput()) {
        if (this.input.type === 'password') {
          this.input.type = 'text';
          this.showPassword = true;
        } else {
          this.input.type = 'password';
          this.showPassword = false;
        }

        this.input.focus();
      }
    },
    clearInput() {
      this.inputInstance.$el.value = '';
      this.inputInstance.$emit('input', '');
      this.setValue('');
    },
    setValue(value) {
      this.value = value;
    },
  },
  mounted() {
    // eslint-disable-next-line
    this.input = this.$el.querySelectorAll('input, textarea, select, .re-file')[0];

    if (!this.input) {
      this.$destroy();

      throw new Error('Missing input/select/textarea inside re-input-container');
    }
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/input.scss";
@import "@/assets/scss/components/buttons.scss";
</style>
