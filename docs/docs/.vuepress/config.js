const path = require("path");
const glob = require("glob")

const baseContext = process.env.CONTEXT || 'docs/'

function getSidebarChildren(location, replacement) {
    if (!replacement) {
        replacement = location
    }
    return glob.sync(
        location + '/**/*.md').map(
            f => f.replace(replacement + '/','')).filter(s => s.toLowerCase().indexOf("readme.md") == -1
        )
}

module.exports = {
  dest: 'site',
  title: 'biome-text',
  description: 'biome-text documentation',
  head: [
    ['link', { rel: "shortcut icon", href: "/favicon.ico"}],
  ],
  base: `/${baseContext}`,
  plugins: [
    '@goy/svg-icons',
    '@vuepress/back-to-top'
  ],
  themeConfig: {
    sidebarDepth: 1,
    displayAllHeaders: false,
    searchPlaceholder: 'Search',
    nav: [
      { text: 'API', link: '/api/'},
      { text: 'Documentation', link: '/documentation/'},
      { text: 'Github', link: 'https://github.com/recognai/biome-text' },
      { text: 'Recognai', link: 'https://recogn.ai' },
    ],
    sidebar: {
      '/api/': [{
          title: 'API',
          children: getSidebarChildren('docs/api'),
          collapsable: false,
      }],
      '/documentation/': [
      {
          title: 'Get started',
          children: ['', 'concepts.md'],
          collapsable: false
      },
      { 
          title: 'Tutorials', 
          children:getSidebarChildren('docs/documentation/tutorials', 'docs/documentation'),
          collapsable: false
      },
      {
          title: 'User Guides',
          children:getSidebarChildren('docs/documentation/user-guides', 'docs/documentation'), 
          collapsable: false
      },
      {
          title: 'Community', 
          children:getSidebarChildren('docs/documentation/community', 'docs/documentation'), 
          collapsable: false
      }]
    },
    algolia: {
      apiKey: '4f8d6b27d633951bde8c33e391ea6a4d',
      indexName: 'recogn_biome-text'
    },
    plugins: ['@vuepress/active-header-links'],
  }
}