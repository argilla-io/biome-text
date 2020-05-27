<template>
  <div class="select" v-click-outside="close">
    <p @click="showSelect = !showSelect">{{selectName}}</p>
    <transition name="fade" appear>
      <ul v-if="showSelect" class="select__options">
        <li v-for="option in options" :key="option" :class="{'selected' : option === selected}">
          <a href="#" v-if="option" @click="selectOption(option)">{{option}}</a>
        </li>
      </ul>
    </transition>
  </div>
</template>

<script>
export default {
  data: () => ({
    showSelect: false,
  }),
  props: {
    selectName: {
      type: String
    },
    options: {
      type: Array,
      default: () => ['1', '2', '3']
    },
    selected: {
      type: String      
    }
  },
  methods: {
    selectOption (option) {
      this.$emit('onselect', option);
      this.showSelect = false;
    },
    close () {
      this.showSelect = false
    }
  }
};
</script>

<style lang="scss" scoped>
@import "@/assets/scss/components/select.scss";
</style>
