<template>
  <span class="nav-item" v-if="options && options.length > 0">
    Version:
    <select v-model="selected" @change="onChange">
      <option v-for="option in options" :value="option">
        {{ option }}
      </option>
    </select>
  </span>
</template>

<script>
import Axios from 'axios';
export default {
  name: 'Versions',

  data() {
    return {
      selected: undefined,
      options: []
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
    onChange(event) {
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