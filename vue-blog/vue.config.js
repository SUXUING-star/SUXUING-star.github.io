const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  
  publicPath: process.env.NODE_ENV === 'production'
    ? '/SUXUING-star.github.io/'
    : '/',
  
  configureWebpack: {
    resolve: {
      fallback: {
        "path": require.resolve("path-browserify")
      }
    }
  },

  chainWebpack: config => {
    // Markdown loader
    config.module
      .rule('markdown')
      .test(/\.md$/)
      .use('frontmatter-markdown-loader')
      .loader('frontmatter-markdown-loader')
      .end()

    // 重新配置图片规则
    config.module
      .rule('images')
      .test(/\.(png|jpe?g|gif|webp|svg)(\?.*)?$/)
      .type('asset/resource')  // 使用 webpack5 的资源模块
      .set('generator', {
        filename: 'img/[name].[hash:8][ext]'  // 输出路径格式
      })
  }
})