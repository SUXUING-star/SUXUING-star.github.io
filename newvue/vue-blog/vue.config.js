const { defineConfig } = require('@vue/cli-service');
const CompressionPlugin = require('compression-webpack-plugin');

module.exports = defineConfig({
  transpileDependencies: true,

  // 动态设置 publicPath，避免硬编码
  publicPath: process.env.NODE_ENV === 'production'
    ? process.env.PUBLIC_PATH || '/SUXUING-star.github.io/'
    : '/',

    configureWebpack: config => {
      // 生产环境特定配置
      if (process.env.NODE_ENV === 'production') {
        // 分包策略优化
        config.optimization = {
          ...config.optimization,
          splitChunks: {
            chunks: 'all',
            minSize: 20000,
            maxSize: 250000,
            cacheGroups: {
              vendor: {
                name: 'chunk-vendors',
                test: /[\\/]node_modules[\\/]/,
                priority: 10,
                chunks: 'initial'
              },
              common: {
                name: 'chunk-common',
                minChunks: 2,
                priority: 5,
                reuseExistingChunk: true
              }
            }
          }
        };
      }
  
      return {
        resolve: {
          fallback: {
            "path": require.resolve("path-browserify")
          }
        },
        plugins: [
          new CompressionPlugin({
            test: /\.(js|css|html|svg)$/i,
            threshold: 10240,
            deleteOriginalAssets: false,
            algorithm: 'gzip',
            minRatio: 0.8
          })
        ]
      };
    },

  chainWebpack: config => {
    // Markdown loader configuration
    config.module
      .rule('markdown')
      .test(/\.md$/)
      .use('frontmatter-markdown-loader')
      .loader('frontmatter-markdown-loader')
      .end();

    // 重新配置图片规则，使用 webpack5 的资源模块
    config.module
      .rule('images')
      .test(/\.(png|jpe?g|gif|webp|svg)(\?.*)?$/)
      .type('asset/resource')
      .set('generator', {
        filename: 'img/[name].[hash:8][ext]'
      })
      .end();

    // 生产环境优化
    if (process.env.NODE_ENV === 'production') {
      config.plugins.delete('prefetch');
      config.performance
        .hints('warning')
        .maxEntrypointSize(500000)
        .maxAssetSize(500000);
    }
  }
});