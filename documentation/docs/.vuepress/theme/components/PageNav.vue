<template>
  <div
    v-if="prev || next"
    class="page-nav"
  >
    <p class="inner">
      <span
        v-if="prev"
        class="page-nav__button prev"
      >
        <a
          v-if="prev.type === 'external'"
          class="prev"
          :href="prev.path"
          target="_blank"
          rel="noopener noreferrer"
        >
          <span class="page-nav__button__icon">
            <vp-icon color="#4A4A4A" name="chev-left" size="18px"/>
          </span>
          {{ prev.title || prev.path }}

          <OutboundLink />
        </a>

        <RouterLink
          v-else
          class="prev"
          :to="prev.path"
        >
          <span class="page-nav__button__icon">
            <vp-icon color="#4A4A4A" name="chev-left" size="18px"/>
          </span>
          {{ prev.title || prev.path }}
        </RouterLink>
      </span>

      <span
        v-if="next"
        class="page-nav__button next"
      >
        <a
          v-if="next.type === 'external'"
          :href="next.path"
          target="_blank"
          rel="noopener noreferrer"
        >
          {{ next.title || next.path }}
          <span class="page-nav__button__icon">
            <vp-icon color="#4A4A4A" name="chev-right" size="18px"/>
          </span>
          <OutboundLink />
        </a>

        <RouterLink
          v-else
          :to="next.path"
        >
          {{ next.title || next.path }}
          <span class="page-nav__button__icon">
            <vp-icon color="#4A4A4A" name="chev-right" size="18px"/>
          </span>
        </RouterLink>
      </span>
    </p>
  </div>
</template>

<script>
import { resolvePage } from '@vuepress/theme-default/util'
import isString from 'lodash/isString'
import isNil from 'lodash/isNil'

export default {
  name: 'PageNav',

  props: ['sidebarItems'],

  computed: {
    prev () {
      return resolvePageLink(LINK_TYPES.PREV, this)
    },

    next () {
      return resolvePageLink(LINK_TYPES.NEXT, this)
    }
  }
}

function resolvePrev (page, items) {
  return find(page, items, -1)
}

function resolveNext (page, items) {
  return find(page, items, 1)
}

const LINK_TYPES = {
  NEXT: {
    resolveLink: resolveNext,
    getThemeLinkConfig: ({ nextLinks }) => nextLinks,
    getPageLinkConfig: ({ frontmatter }) => frontmatter.next
  },
  PREV: {
    resolveLink: resolvePrev,
    getThemeLinkConfig: ({ prevLinks }) => prevLinks,
    getPageLinkConfig: ({ frontmatter }) => frontmatter.prev
  }
}

function resolvePageLink (
  linkType,
  { $themeConfig, $page, $route, $site, sidebarItems }
) {
  const { resolveLink, getThemeLinkConfig, getPageLinkConfig } = linkType

  // Get link config from theme
  const themeLinkConfig = getThemeLinkConfig($themeConfig)

  // Get link config from current page
  const pageLinkConfig = getPageLinkConfig($page)

  // Page link config will overwrite global theme link config if defined
  const link = isNil(pageLinkConfig) ? themeLinkConfig : pageLinkConfig

  if (link === false) {
    return
  } else if (isString(link)) {
    return resolvePage($site.pages, link, $route.path)
  } else {
    return resolveLink($page, sidebarItems)
  }
}

function find (page, items, offset) {
  const res = []
  flatten(items, res)
  for (let i = 0; i < res.length; i++) {
    const cur = res[i]
    if (cur.type === 'page' && cur.path === decodeURIComponent(page.path)) {
      return res[i + offset]
    }
  }
}

function flatten (items, res) {
  for (let i = 0, l = items.length; i < l; i++) {
    if (items[i].type === 'group') {
      flatten(items[i].children || [], res)
    } else {
      res.push(items[i])
    }
  }
}
</script>

<style lang="stylus">

.page-nav
  padding-top 1rem
  padding-bottom 0
  max-width: 740px;
  margin: 0 auto;
  padding: 1rem 2.5rem 0 2.5rem;
  @media (max-width: $MQMobileNarrow)
    padding: 1.5rem;
  &__button
    border: 1px solid $borderColor
    border-radius: 3px
    font-family: $secondaryFontFamily
    &:hover
      border-color: $textColor !important
    a
      display: flex
      padding: 0.5em
      color: $textColor !important
      font-weight: 600
      font-size: 0.75rem
    &__icon
      display: flex
      align-items: center
    @media (max-width: $MQMobileNarrow)
      border: none
      a
        padding: 0
  .inner
    min-height 2rem
    margin-top 0
    border-top 1px solid $borderColor
    padding-top 1rem
    overflow auto // clear float
  .prev
    float: left
    .page-nav__button__icon
      float: left
      margin-right: 1.5rem
  .next
    float right
    .page-nav__button__icon
      float: right
      margin-left: 1.5rem
</style>