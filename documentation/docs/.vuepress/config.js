const path = require("path");
const { nav, sidebar } = require("vuepress-bar")(`${__dirname}/..`, options = { maxLevel: 1, addReadMeToFirstGroup: false, collapsable: false, });
const baseContext = process.env.CONTEXT || 'docs'

module.exports = {
  dest: 'site',
  title: 'biome-text',
  description: 'biome-text documentation',
  base: `/${baseContext}`,
  plugins: [
    '@goy/svg-icons',
    '@vuepress/back-to-top'
  ],
  themeConfig: {
    nav: [
      ...nav,
      { text: 'Github', link: 'https://github.com/recognai/biome-text' },
      { text: 'Recognai', link: 'https://recogn.ai' },
    ],
    displayAllHeaders: true,
    sidebar,
    plugins: ['@vuepress/active-header-links'],
    // logo: '/assets/img/recognai.png',
  }
}