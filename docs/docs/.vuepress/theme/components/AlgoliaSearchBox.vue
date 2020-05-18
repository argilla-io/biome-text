<template>
  <form
    id="search-form"
    class="algolia-search-wrapper search-box"
    role="search"
  >
    <input
      id="algolia-search-input"
      class="search-query"
      :placeholder="placeholder"
    >
  </form>
</template>

<script>
export default {
  name: 'AlgoliaSearchBox',

  props: ['options'],

  data () {
    return {
      placeholder: undefined
    }
  },

  watch: {
    $lang (newValue) {
      this.update(this.options, newValue)
    },

    options (newValue) {
      this.update(newValue, this.$lang)
    }
  },

  mounted () {
    this.initialize(this.options, this.$lang)
    this.placeholder = this.$site.themeConfig.searchPlaceholder || ''
  },

  methods: {
    initialize (userOptions, lang) {
      Promise.all([
        import(/* webpackChunkName: "docsearch" */ 'docsearch.js/dist/cdn/docsearch.min.js'),
        import(/* webpackChunkName: "docsearch" */ 'docsearch.js/dist/cdn/docsearch.min.css')
      ]).then(([docsearch]) => {
        docsearch = docsearch.default
        const { algoliaOptions = {}} = userOptions
        docsearch(Object.assign(
          {},
          userOptions,
          {
            inputSelector: '#algolia-search-input',
            // #697 Make docsearch work well at i18n mode.
            algoliaOptions: Object.assign({
              'facetFilters': [`lang:${lang}`].concat(algoliaOptions.facetFilters || [])
            }, algoliaOptions),
            handleSelected: (input, event, suggestion) => {
              const { pathname, hash } = new URL(suggestion.url)
              const routepath = pathname.replace(this.$site.base, '/')
              this.$router.push(`${routepath}${hash}`)
            }
          }
        ))
      })
    },

    update (options, lang) {
      this.$el.innerHTML = '<input id="algolia-search-input" class="search-query">'
      this.initialize(options, lang)
    }
  }
}
</script>

<style lang="stylus">
.algolia-search-wrapper
  & > span
    vertical-align middle
  .algolia-autocomplete
    line-height normal
    .ds-dropdown-menu
      background-color #fff
      border 2px solid $accentColor
      border-radius 0
      font-size 16px
      margin 6px 0 0
      padding 0
      text-align left
      [class^=ds-dataset-]
        border-radius 0
        padding 0
        border 0
      &:before
        display none
      .ds-suggestions
        margin-top 0
    .algolia-docsearch-suggestion--highlight
      padding 0
      color $accentColor !important
      background #F0E7FF !important
      box-shadow none !important
    .algolia-docsearch-suggestion
      border-color $accentColor
      padding 0
      .algolia-docsearch-suggestion--category-header
        padding 0.9em;
        margin-top 0
        background $accentColor
        color #fff
        font-weight 600
        font-size 15px
        border none
        .algolia-docsearch-suggestion--highlight
          color $accentColor
          background #F0E7FF
          box-shadow none
      .algolia-docsearch-suggestion--wrapper
        padding 0.9em
        &:hover
          background: #FCFCFC
        &:after
          content ""
          border-bottom 1px solid $accentColor
          position absolute 
          left 1em
          right 1em
          bottom 0
      .algolia-docsearch-suggestion--title
        float none
        font-weight 600
        margin-bottom 0.5em
        color $textColor
        text-align left
      .algolia-docsearch-suggestion--subcategory-column
        float none
        text-align left
        width 100% !important
        vertical-align top
        padding 0
        border none
        &:before
          display none
        &:after
          display none
      .algolia-docsearch-suggestion--subcategory-column-text
        width 100%
        color #555
        margin-bottom 0.5em
        &:before
          content "#"
          display inline-block
          color $accentColor
          margin-right 0.5em
      .algolia-docsearch-suggestion--content
        padding-bottom 0.5em
        &:before
          display none
    .algolia-docsearch-footer
      border-color $accentColor
    .ds-cursor .algolia-docsearch-suggestion--content
      color $textColor
      background-color: transparent !important
    .algolia-docsearch-footer--logo
      padding 0.9em

@media (min-width: $MQMobile)
  .algolia-search-wrapper
    .algolia-autocomplete
      .algolia-docsearch-suggestion
        .algolia-docsearch-suggestion--subcategory-column
          float none
          width 100%
          display block
          padding 0
        .algolia-docsearch-suggestion--content
          float none
          width 100%
          vertical-align top
          display block
          padding 0
        .ds-dropdown-menu
          min-width 515px !important
@media (max-width: $MQMobile)
  .algolia-search-wrapper
    .ds-dropdown-menu
      min-width calc(100vw - 4rem) !important
      max-width calc(100vw - 4rem) !important
    .algolia-docsearch-suggestion--subcategory-column
      padding 0 !important
      background white !important
    .algolia-docsearch-suggestion--subcategory-column-text:after
      content " > "
      font-size 10px
      line-height 14.4px
      display inline-block
      width 5px
      margin -3px 3px 0
      vertical-align middle

</style>