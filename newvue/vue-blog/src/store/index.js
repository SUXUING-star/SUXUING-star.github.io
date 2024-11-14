// src/store/index.js
import { createStore } from 'vuex'

// 使用 webpack 的 require.context 来导入 Markdown 文件
const markdownFiles = require.context('../posts/', true, /\.md$/)

// 格式化日期
function formatDate(dateString) {
  const date = new Date(dateString)
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  })
}

// 处理图片路径
function getImageUrl(imagePath) {
  if (!imagePath) return ''
  if (imagePath.startsWith('http')) {
    return imagePath
  }
  
  try {
    // 使用 require 动态导入图片
    return require(`../posts/images/${imagePath}`)
  } catch (e) {
    console.warn(`Image not found: ${imagePath}`)
    return ''
  }
}

// 处理 Markdown 内容中的图片路径
function processMarkdownImages(content) {
  return content.replace(
    /!\[(.*?)\]\((.*?)\)/g,
    (match, alt, path) => {
      // 忽略外部链接
      if (path.startsWith('http')) {
        return match
      }
      // 处理本地图片路径
      try {
        const imageUrl = require(`../posts/images/${path}`)
        return `![${alt}](${imageUrl})`
      } catch (e) {
        console.warn(`Image not found in markdown: ${path}`)
        return match
      }
    }
  )
}

// 处理 Markdown 文件
function processMarkdownFiles() {
  return markdownFiles.keys().map((path, index) => {
    // 从文件路径中提取文件名作为URL slug
    const slug = path.replace(/^\.\//, '').replace(/\.md$/, '')
    
    // 获取文件内容
    const { attributes, html } = markdownFiles(path)
    
    return {
      id: index + 1,
      slug,
      title: attributes.title,
      date: formatDate(attributes.date), // 格式化日期
      summary: attributes.summary,
      coverImage: getImageUrl(attributes.coverImage),
      content: processMarkdownImages(html)
    }
  })
}

const store = createStore({
  state() {
    return {
      posts: processMarkdownFiles()
    }
  },
  getters: {
    getPostById: (state) => (id) => {
      return state.posts.find(post => post.id === parseInt(id))
    },
    getPostBySlug: (state) => (slug) => {
      return state.posts.find(post => post.slug === slug)
    },
    getAllPosts: (state) => {
      return state.posts
    }
  },
  mutations: {
    UPDATE_POST(state, { id, post }) {
      const index = state.posts.findIndex(p => p.id === id)
      if (index !== -1) {
        state.posts[index] = { ...state.posts[index], ...post }
      }
    }
  },
  actions: {
    updatePost({ commit }, payload) {
      commit('UPDATE_POST', payload)
    }
  }
})

export default store