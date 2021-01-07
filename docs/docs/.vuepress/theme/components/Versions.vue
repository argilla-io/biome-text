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
      let res = await Axios.get(
        'https://api.github.com/repos/recognai/biome-text/git/trees/gh-pages'
      );
      const versionsNode = res.data.tree.find(e => {
        return e.path.toLowerCase() === 'versions';
      });
      res = await Axios.get(versionsNode.url);
      this.options = res.data.tree.map(e => {
        return e.path ;
      });
      this.options.sort();
      this.options.unshift('master');
      const paths = window.location.pathname.split('/');
      if (paths[2] === 'versions') {
        this.selected = paths[3];
      } else {
        this.selected = 'master';
      }
    } catch (ex) {}
  },
  methods: {
    onChange(option) {
      this.showOptions = false;
      this.selected = option;
      const targetVersionsPath =
        this.selected === 'master' ? '/' : `/versions/${this.selected}/`;
      const paths = window.location.pathname.split('/')
      if (paths[2] === 'versions') {
        window.location.pathname =
          paths.slice(0,2).join('/') +
          targetVersionsPath +
          paths.slice(4).join('/')
      } else {
        window.location.pathname =
          paths.slice(0,2).join('/') +
          targetVersionsPath +
          paths.slice(2).join('/')

      }
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
  &__select
    background: transparent
    min-height: 30px
    padding: 0.5em
    color: $textColor
    font-size: 15px
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
    padding: 0em 0.5em
    font-size: 14px
    font-size: 15px
    text-align: left
    a
      color: $textColor
      &:hover, &:focus, &.active
        color: $accentColor

</style>

