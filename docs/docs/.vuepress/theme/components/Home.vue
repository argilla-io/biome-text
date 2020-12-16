<template>
<div>
  <div class="home__nav">
    <RouterLink to="/">
      <img
        class="home__nav__logo"
        v-if="data.navImage"
        :src="$withBase(data.navImage)"
      >
    </RouterLink>
  </div>
  <main
    class="home"
    aria-labelledby="main-title"
  >
    <header class="hero">
      <img
        v-if="data.heroImage"
        :src="$withBase(data.heroImage)"
        :alt="data.heroAlt || 'hero'"
      >

      <h1
        v-if="data.heroText !== null"
        id="main-title"
      >
        {{ data.heroText || $title || 'Hello' }}<span>{{data.heroSubText }}</span>
      </h1>

      <p
        v-if="data.tagline !== null"
        class="description"
      >
        {{ data.tagline || $description || 'Welcome to your VuePress site' }}
      </p>

      <p
        v-if="data.actionText && data.actionLink"
        class="action"
      >
        <NavLink
          class="action-button"
          :item="actionLink"
        />
      </p>
    </header>

    <div
      v-if="data.features && data.features.length"
      class="features"
    >
      <div
        v-for="(feature, index) in data.features"
        :key="index"
        class="feature"
      >
        <h2>{{ feature.title }}</h2>
        <p>{{ feature.details }}</p>
        <span class="feature__images">
          <img
            v-if="feature.img1"
            :src="$withBase(feature.img1)"
          >
          <img
            v-if="feature.img2"
            :src="$withBase(feature.img2)"
          >
          <img
            v-if="feature.img3"
            :src="$withBase(feature.img3)"
          >
        </span>
      </div>
    </div>

    <Content class="theme-default-content custom" />

    <div
      class="footer"
    >
      <div>
        {{ data.footer }}
        <a href="https://recogn.ai" target="_blank"><img width="70px" :src="$withBase('/assets/img/recognai.png')" /></a>
      </div>
    </div>
  </main>

<svg class="home__bg" width="494px" height="487px" viewBox="0 0 494 487" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
    <g id="*-Documentation" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
        <g id="bg" transform="translate(-967.000000, 0.000000)" stroke="#4C1DBF">
            <g id="Page-1" transform="translate(1283.285018, 114.169241) scale(1, -1) rotate(-222.000000) translate(-1283.285018, -114.169241) translate(905.285018, -340.330759)">
                <path d="M190.848191,92.6665641 C190.848191,192.936685 213.213577,625.695092 235.586289,736.523335 C254.765514,831.56423 285.912986,889.573385 330.651084,889.573385 C364.200995,889.573385 384.068724,834.404885 403.343168,720.689711 C425.715879,588.752345 436.900404,182.385682 436.900404,97.9420653 C436.900404,48.4228362 434.966734,0 317.536552,0 C200.099045,0 190.848191,42.0900989 190.848191,92.6665641 Z" id="Stroke-1"></path>
                <path d="M462.669807,123.072945 C430.876905,211.06309 314.908331,598.02194 301.004923,702.481229 C289.087189,792.060095 300.264439,852.999143 342.737252,867.408306 C374.592783,878.218748 410.946475,836.208684 465.303866,742.63481 C528.370228,634.062698 667.827973,281.070342 694.60327,206.971687 C710.30447,163.519279 723.824742,120.402466 612.328543,82.569488 C500.836027,44.7365105 478.706251,78.6922951 462.669807,123.072945 Z" id="Stroke-3"></path>
                <path d="M522.165599,265.889246 C484.284461,338.587617 340.934261,662.14445 319.206142,752.302566 C300.572019,829.62022 306.700936,885.333623 346.978259,904.95665 C377.187174,919.671243 415.917499,888.385777 476.23195,814.402289 C546.21606,728.556455 709.802763,438.844613 741.702669,377.626624 C760.410634,341.725445 776.966094,305.770719 671.234891,254.266074 C565.503688,202.761429 541.272313,229.224136 522.165599,265.889246 Z" id="Stroke-5"></path>
                <path d="M7.4784994,296.640189 C36.7838113,378.827884 184.591094,726.457068 238.303966,810.212619 C284.362025,882.038214 330.995856,919.720195 373.625119,905.548919 C405.598912,894.923135 408.403955,843.411958 393.53723,744.099005 C376.28991,628.87232 268.166596,292.24663 243.482222,223.032939 C229.006727,182.4433 213.014294,143.364503 101.111555,180.55831 C-10.791185,217.752117 -7.30702698,255.184666 7.4784994,296.640189 Z" id="Stroke-7"></path>
                <path d="M222.393346,293.866117 C222.393346,371.203174 244.758733,704.97758 267.127782,790.456697 C286.31067,863.760269 317.458142,908.500479 362.196239,908.500479 C395.746151,908.500479 415.61388,865.948649 434.888324,778.245395 C457.261035,676.485733 468.445559,363.064689 468.445559,297.935359 C468.445559,259.74241 466.511889,222.393346 349.078045,222.393346 C231.644201,222.393346 222.393346,254.857889 222.393346,293.866117 Z" id="Stroke-9"></path>
            </g>
        </g>
    </g>
</svg>
</div>
</template>

<script>
import NavLink from '@theme/components/NavLink.vue'

export default {
  name: 'Home',

  components: { NavLink },

  computed: {
    data () {
      return this.$page.frontmatter
    },

    actionLink () {
      return {
        link: this.data.actionLink,
        text: this.data.actionText
      }
    }
  },

  mounted () {
    document.querySelector('.global-ui').classList.add('hidden')
  }
}
</script>

<style lang="stylus">
.hidden
  display none
.home
  padding $navbarHeight 2rem 0
  padding-top 0
  max-width $homePageWidth
  margin 0px auto
  display block
  position relative
  z-index 1
  &__bg
    width 500px
    height 536px
    position absolute
    right 0
    top 0
    path
      transform rotateY(0) translateY(0)
      transform-origin center
      animation animate 10s infinite
      &:nth-child(1)
        animation-delay 1s
      &:nth-child(2)
        animation-delay 3s
      &:nth-child(3)
        animation-delay 4s
      &:nth-child(4)
        animation-delay 2s
      &:nth-child(5)
        animation-delay 0


    // for num in (1..5)
    //   path:nth-child({num})
    //     animation animate 8s infinite
    //     stroke red
    //     animation-delay "calc(0.5 * %s)" % num
  &__nav
    z-index 1
    border-bottom 1px solid $borderColor
    display block
    background #fff
    position relative
    &__logo
      max-width 120px
      padding 0.5em;
  .hero
    text-align center
    img
      max-width: 100%
      max-height 280px
      display block
      margin 3rem auto 1.5rem
    h1
      font-size 3rem
      font-family $primaryFontFamily
      span
        font-weight lighter
        font-family 'Basis Grotesque Pro Light'
    h1, .description, .action
      margin 1.8rem auto
    .description
      max-width 400px
      font-size 1.8rem
      line-height 1.3
      color $textColorLight
      font-family 'Basis Grotesque Pro Light'
    .action-button
      display inline-block
      font-size 1.2rem
      color #fff
      background-color $accentColor
      padding 0.6rem 1.6rem
      border-radius 4px
      transition background-color .1s ease
      box-sizing border-box
      min-width 200px
      &:hover
        background-color darken($accentColor, 10%)
  .features
    padding 1.2rem 0
    margin-top 2.5rem
    display flex
    flex-wrap wrap
    align-items flex-start
    align-content stretch
    justify-content space-between
  .feature
    flex-grow 1
    flex-basis 30%
    max-width 30%
    h2
      font-size 1.4rem
      font-weight bolder
      border-bottom none
      padding-bottom 0
      color #4C10BC
    p
      color $textColor
    &__images
      display block
      img
        max-width 90px
        max-height 30px
        margin-right 1em
        vertical-align middle
  .footer
    padding 2.5rem
    // border-top 1px solid $borderColor
    text-align center
    color lighten($textColor, 25%)
    font-size 12px
    & > div
      margin: auto
      display flex
      align-items center
      width 160px
    img
     margin-left 1em

@media (max-width: $MQMobile)
  .home
    &__bg
      opacity 0.3
    .features
      flex-direction column
    .feature
      max-width 100%
      padding 0 2.5rem

@media (max-width: $MQMobileNarrow)
  .home
    padding-left 1.5rem
    padding-right 1.5rem
    .hero
      img
        max-height 210px
        margin 2rem auto 1.2rem
      h1
        font-size 2rem
      h1, .description, .action
        margin 1.2rem auto
      .description
        font-size 1.2rem
      .action-button
        font-size 1rem
        padding 0.6rem 1.2rem
    .feature
      h2
        font-size 1.25rem

@keyframes animate
  0%
   transform rotateY(0) translateY(0)
  50%
   transform rotateY(60deg) translateY(0);
  100%
   transform rotateY(0) translateY(0)
</style>
