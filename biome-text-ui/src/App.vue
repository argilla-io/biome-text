<template>
  <div id="app">
      <!-- <div id="nav">
      <router-link :to="{name: 'home', params: {exploreAs: 'feedback'} }">Feedback</router-link>|
      <router-link :to="{name: 'home', params: {exploreAs:'explore'} }">Explore</router-link>
    </div> -->
      <main>
        <router-view></router-view>
      </main>
  </div>
</template>

<style lang="scss">
@import "@/assets/scss/base/base.scss";
</style>
