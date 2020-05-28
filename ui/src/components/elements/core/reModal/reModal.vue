<template>
  <transition name="modal" v-if="modalVisible" appear>
    <div id="modal-template">
      <div class="modal-mask" v-click-outside="onClickOutside">
        <div class="modal-wrapper" :class="modalPosition">
          <div :class="['modal-container', modalClass]">
            <p class="modal__title" v-if="!modalCustom">
              <span class="state" :class="modalClass === 'modal-info' ? 'succeeded': 'failed'"></span>
              {{modalTitle}}
            </p>
            <div v-if="!modalCustom">
            </div>
            <slot></slot>
            <re-button v-if="modalCloseButton" class="modal-close" @click="$emit('close-modal')">
              <svgicon name="cross" width="10" height="auto"></svgicon>
            </re-button>
          </div>
        </div>
      </div>
    </div>
  </transition>
</template>

<script>
import '@/assets/iconsfont/cross';
import reButton from '@/components/elements/core/reButton/reButton';

export default {
  name: 're-modal',
  data: () => ({
  }),
  props: {
    modalCloseButton: {
      type: Boolean,
      default: true,
    },
    modalVisible: {
      type: Boolean,
      default: true,
    },
    modalCustom: {
      type: Boolean,
      default: false,
    },
    modalClass: {
      type: String,
      default: 'modal-info',
    },
    modalTitle: {
      type: String,
      default: undefined,
    },
    preventBodyScroll: {
      type: Boolean,
      default: false,
    },
    messages: {
      type: Array,
      default: () => [],
    },
    modalPosition: {
      type: String,
      default: 'modal-bottom',
    },
  },
  components: {
    reButton,
  },
  methods: {
    onClickOutside() {
      this.$emit('close-modal');
    },
  },
  updated() {
    if (this.preventBodyScroll) {
      if (this.modalVisible) {
        document.body.classList.add('--fixed');
      } else {
        document.body.classList.remove('--fixed');
      }
    }
  },
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/modal.scss";
@import "@/assets/scss/components/state-icons.scss";
</style>
