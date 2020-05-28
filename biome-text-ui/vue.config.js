const elasticsearchEndpoint = process.env.ELASTICSEARCH || 'http://localhost:9200';
const publicPath = process.env.BASE_URL || '/';

module.exports = {
  publicPath,
  outputDir: `../src/biome/text/ui/webapp${publicPath}`,
  runtimeCompiler: true,
  css: {
    modules: true,
    loaderOptions: {
      // pass options to sass-loader
      sass: {
        // @/ is an alias to src/
        // so this assumes you have a file named `src/variables.scss`
        data: '@import "@/assets/scss/main.scss";',
      },
    },
  },
  chainWebpack: (config) => {
    config.module
      .rule('vue')
      .use('vue-loader')
      .loader('vue-loader')
      .tap(options => options);
  },
  devServer: {
    proxy: {
      '.*/elastic': {
        target: elasticsearchEndpoint,
        pathRewrite: { '.*/elastic': '' },
      },
    },
  },
};
