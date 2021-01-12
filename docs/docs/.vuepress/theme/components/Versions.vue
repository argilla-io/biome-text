<template>
  <span class="nav-versions" v-if="options && options.length > 0">
    <div class="nav-versions__select" v-model="selected" @click="showOptions = true"><strong>{{selected}}</strong></div>
    <div class="nav-versions__options__container" v-if="showOptions">
      <ul class="nav-versions__options">
        <li class="nav-versions__option" @click="onChange(option)" v-for="option in options" :value="option">
          <a :class="option === selected ? 'active' : ''" href="#">{{ option }}</a>
        </li>
      </ul>
    </div>
  </span>
</template>

<script>
import Axios from 'axios';
export default {
  name: 'Versions',

  data() {
    return {
      selected: undefined,
      options: [],
      showOptions: false,
    };
  },
  created: async function() {
    try {
      // This hardcoded url will be problematic if we want to move away from github pages ...
      let res = await Axios.get(
        'https://raw.githubusercontent.com/recognai/biome-text/gh-pages/versions.txt'
      );
      this.options = res.data.split('\n')
        .filter((e) => {return e !== ""})
        .map((e) => {return e.trim()})

      this.selected = window.location.pathname.split('/')[2];
    } catch (ex) {}
  },
  methods: {
    onChange(option) {
      this.showOptions = false;
      this.selected = option;
      const targetVersionPath = `/${this.selected}/`;
      const paths = window.location.pathname.split('/');
      window.location.pathname =
        paths.slice(0,2).join('/') +
        targetVersionPath +
        paths.slice(3).join('/')
    }
  }
};
</script>
<style lang="stylus">

.nav-versions
  display: block
  margin: auto
  text-align: center
  position: relative
  z-index: 1
  &__select
    background: transparent
    min-height: 30px
    padding: 0.5em
    color: $textColor
    font-size: 15px
    cursor: pointer
    @media (max-width: $MQMobile)
      font-size: 16px
    &::after
      content: ''
      border-left: 4px solid transparent
      border-right: 4px solid transparent
      border-top: 6px solid $arrowBgColor
      border-bottom: 0
      display: inline-block
      margin-left: 0.5em
  &__options
    background: white
    border: 1px solid $borderColor
    display: inline-block
    width: auto
    list-style: none
    border-radius: 4px
    &__container
      position: absolute
      top: 2em
      left: 0
      right: 0
      margin: auto !important
  &__option
    padding: 0.2em 1em
    font-size: 15px
    text-align: left
    @media (max-width: $MQMobile)
      padding: 0.5em 2em
    a
      color: $textColor
      &:hover, &:focus, &.active
        color: $accentColor

</style>
