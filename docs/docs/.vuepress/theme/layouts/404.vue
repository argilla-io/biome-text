<template>
  <div class="theme-container error">
    <div class="home__nav">
      <img
        class="home__nav__logo"
        :src="$withBase('/assets/img/biome.svg')"
      >
    </div>
    <div class="theme-default-content">
      <h1 class="error__title">Error 404</h1>
      <p class="error__subtitle">{{ getMsg() }}</p>
      <p class="error__text">Oh no!</br>
      It looks like we can’t meet right now</p>
      <RouterLink to="/">
        Take me home
      </RouterLink>
    </div>
    <div
      class="footer"
    >
      <div>
        Built by 
        <img width="70px" :src="$withBase('/assets/img/recognai.png')" />
      </div>  
    </div>
  </div>
</template>

<script>
const msgs = [
  `There's nothing here.`,
  `How did we get here?`,
  `Page not Found`,
  `Looks like we've got some broken links.`
]

export default {
  methods: {
    getMsg () {
      return msgs[Math.floor(Math.random() * msgs.length)]
    }
  }
}
</script>
<style lang="stylus" scoped>
.theme-default-content
  margin-left auto !important
.home
  &__nav
    z-index 1
    border-bottom 1px solid $borderColor
    display block
    background #fff
    position relative
    &__logo
      max-width 120px
      padding 0.5em
.error
  position relative
  min-height 120vh
  background url('./error-bg.svg') center bottom no-repeat 
  background-size 100%;
  &__title
    display: inline-block
    margin-right 1em
    margin-top 0 !important
    padding-top: 0 !important
    font-size 2em
  &__subtitle
    display: inline-block
    font-size 2em
    font-weight lighter
    margin-top 0 !important
    margin-bottom 0
    color $textColorLight
  &__text
    font-family $handMadeFontFamily
    color #4C10BC
    font-size: 4.2rem
    line-height: 1em
    margin-top 0
    margin-bottom 0.5em
.footer
  position absolute 
  bottom 2em
  padding 2.5rem 2.5em 0
  text-align center
  color lighten($textColor, 25%)
  font-size 12px
  width 100%
  & > div
    margin: auto
    display flex
    align-items center
    width 126px
  img 
    margin-left 1em
@media (max-width: $MQMobile)
  .error
    &__title
      font-size 1.5em
    &__subtitle
      font-size 1.3em
      display block
    &__text
      font-size 3em 
</style>