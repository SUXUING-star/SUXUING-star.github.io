<!-- PostDetail.vue -->
<template>
  <div class="detail-post" v-if="post">
    <div class="detail-header">
      <h1>{{ post.title }}</h1>
      <div class="detail-meta">
        <time :datetime="post.date">{{ post.date }}</time>
        <div class="meta-divider"></div>
        <span class="detail-word-count">{{ wordCount }} 字</span>
        <div class="meta-divider"></div>
        <span class="detail-reading-time">{{ readingTime }} 分钟阅读</span>
      </div>
    </div>
    
    <div class="detail-container">
      <div class="detail-main">
        <!-- 文章主体内容 -->
        <div class="detail-content markdown-body" v-html="renderedContent" ref="contentRef"></div>
      </div>

      <!-- 右侧导航栏 -->
      <aside class="detail-sidebar">
        <!-- 文章目录 -->
        <div class="sidebar-section toc-section" v-if="tableOfContents">
          <h3 class="sidebar-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" 
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="21" y1="10" x2="7" y2="10"></line>
              <line x1="21" y1="6" x2="3" y2="6"></line>
              <line x1="21" y1="14" x2="3" y2="14"></line>
              <line x1="21" y1="18" x2="7" y2="18"></line>
            </svg>
            目录
          </h3>
          <nav class="toc-nav" v-html="tableOfContents"></nav>
        </div>

        <!-- 阅读进度 -->
        <div class="sidebar-section progress-section">
          <h3 class="sidebar-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 20h9"></path>
              <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path>
            </svg>
            阅读进度
          </h3>
          <div class="progress-bar">
            <div class="progress-inner" :style="{ width: readingProgress + '%' }"></div>
          </div>
          <div class="progress-text">{{ readingProgress }}%</div>
        </div>

        <!-- 最近文章 -->
        <div class="sidebar-section recent-posts">
          <h3 class="sidebar-title">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
            最近文章
          </h3>
          <ul class="recent-posts-list">
            <li v-for="recentPost in recentPosts" :key="recentPost.id" 
                :class="{ active: recentPost.id === post.id }">
              <router-link :to="'/post/' + recentPost.id">
                {{ recentPost.title }}
                <span class="post-date">{{ recentPost.date }}</span>
              </router-link>
            </li>
          </ul>
        </div>
      </aside>
    </div>
  </div>
</template>
  
<script>
import { computed, onMounted, ref, watch, nextTick, onBeforeUnmount } from 'vue'
import { useRoute } from 'vue-router'
import { useStore } from 'vuex'
import { marked } from 'marked'
import Prism from 'prismjs'
import 'prismjs/themes/prism-tomorrow.css'
import 'prismjs/components/prism-markup'
import 'prismjs/components/prism-css'
import 'prismjs/components/prism-javascript'
import 'prismjs/components/prism-typescript'
import 'prismjs/components/prism-python'
import 'prismjs/components/prism-bash'
import 'prismjs/components/prism-json'
import 'prismjs/plugins/line-numbers/prism-line-numbers'
import 'prismjs/plugins/line-numbers/prism-line-numbers.css'

export default {
  name: 'PostDetailPage',
  setup() {
    const route = useRoute()
    const store = useStore()
    const contentRef = ref(null)
    const readingProgress = ref(0)
    const tableOfContents = ref('')

    const post = computed(() => 
      store.state.posts.find(p => p.id === parseInt(route.params.id))
    )

    const wordCount = computed(() => {
      if (!post.value?.content) return 0
      return post.value.content.replace(/\s+/g, '').length
    })

    const readingTime = computed(() => {
      return Math.ceil(wordCount.value / 300)
    })

    const recentPosts = computed(() => {
      return store.state.posts
        .slice()
        .sort((a, b) => new Date(b.date) - new Date(a.date))
        .slice(0, 5)
    })

    const configureMarked = () => {
      const renderer = new marked.Renderer()

      // Generate unique IDs for headers
      renderer.heading = (text, level) => {
        const slug = text.toLowerCase().replace(/\s+/g, '-').replace(/[^\w-]+/g, '')
        return `
          <h${level} id="${slug}">
            ${text}
          </h${level}>
        `
      }

      marked.setOptions({
        renderer,
        gfm: true,
        breaks: true,
        smartLists: true
      })
    }

    const generateTableOfContents = () => {
      if (!contentRef.value) return ''
      
      const headings = contentRef.value.querySelectorAll('h2, h3, h4')
      if (headings.length === 0) return ''

      let toc = '<ul class="toc-list">'
      let levelStack = []

      headings.forEach((heading) => {
        const level = parseInt(heading.tagName.charAt(1))
        const text = heading.textContent
        const id = heading.id

        while (levelStack.length > 0 && levelStack[levelStack.length - 1] >= level) {
          toc += '</ul>'
          levelStack.pop()
        }

        if (levelStack.length === 0 || levelStack[levelStack.length - 1] < level) {
          toc += '<ul>'
          levelStack.push(level)
        }

        toc += `
          <li>
            <a href="#${id}" class="toc-link">
              ${text}
            </a>
          </li>
        `
      })

      while (levelStack.length > 0) {
        toc += '</ul>'
        levelStack.pop()
      }

      return toc
    }

    const processMarkdownContent = (content) => {
      if (!content) return ''
      return marked.parse(content)
    }

    const renderedContent = computed(() => {
      if (!post.value?.content) return ''
      try {
        return processMarkdownContent(post.value.content)
      } catch (err) {
        console.error('Markdown rendering error:', err)
        return '<p>Error rendering content</p>'
      }
    })

    onMounted(() => {
      configureMarked()
      window.addEventListener('scroll', updateReadingProgress)

      nextTick(() => {
        Prism.highlightAll()
        tableOfContents.value = generateTableOfContents()
        updateReadingProgress()
      })
    })

    onBeforeUnmount(() => {
      window.removeEventListener('scroll', updateReadingProgress)
    })

    watch(renderedContent, () => {
      nextTick(() => {
        Prism.highlightAll()
        tableOfContents.value = generateTableOfContents()
        updateReadingProgress()
      })
    })

    const updateReadingProgress = () => {
      if (!contentRef.value) return
      const contentHeight = contentRef.value.offsetHeight
      const scrollPosition = window.scrollY
      const windowHeight = window.innerHeight
      const progress = (scrollPosition / (contentHeight - windowHeight)) * 100
      readingProgress.value = Math.min(Math.max(Math.round(progress), 0), 100)
    }

    return {
      post,
      contentRef,
      tableOfContents,
      readingProgress,
      wordCount,
      readingTime,
      recentPosts,
      renderedContent
    }
  }
}
</script>

  
  <style>
.detail-post {
  max-width: 1600px;  /* 从 1400px 增加到 1600px */
  margin: 0 auto;
  padding: 0.3rem;
}

.detail-header {
  text-align: center;
  max-width: 800px;  /* 从 1000px 增加到 1200px */
  margin: 0 auto 2rem;
  padding: 2rem 1rem;
  background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(238, 242, 247, 0.95) 100%);
  border-radius: 16px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  position: relative;
  overflow: hidden;
}
/* 添加背景装饰 */
.detail-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #42b883, #3aa876);
}
.detail-header h1 {
  font-size: 1.75rem;
  color: #2c3e50;
  margin-bottom: 0.75rem;
  line-height: 1.3;
  font-weight: 600;
}

.detail-meta {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 2rem;
  color: #64748b;
  font-size: 0.9375rem;
  position: relative;
}

.detail-meta::before,
.detail-meta::after {
  content: '';
  height: 1px;
  width: 50px;
  background: #e2e8f0;
  margin: 0 1rem;
}


.detail-reading-time {
  display: flex;
  margin: 0 auto;
  gap: 0.5rem;
  padding: 0.25rem 0.75rem;
  background: rgba(66, 184, 131, 0.1);
  border-radius: 20px;
  color: #42b883;
  font-weight: 500;
}


.detail-container {
  display: grid;
  grid-template-columns: 1fr 280px;  /* 右侧导航栏从 300px 减少到 280px */
  gap: 2rem;
  max-width: 1600px;  /* 从 1200px 增加到 1600px，与 .detail-post 保持一致 */
  margin: 0 auto;
  position: relative;
}
.detail-toc {
  position: sticky;
  top: 5rem;
  height: fit-content;
  background: #ffffff;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  border: 1px solid #eef2f7;
}

.detail-toc-title {
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid #42b883;
}

.detail-toc-content {
  font-size: 0.875rem;
  line-height: 1.6;
}

.detail-toc-content a {
  color: #4a5568;
  text-decoration: none;
  display: block;
  padding: 0.25rem 0;
  transition: color 0.2s ease;
}

.detail-toc-content a:hover {
  color: #42b883;
}

.detail-content {
  background: #ffffff !important;
  padding: 2.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
  border: 1px solid #eef2f7;
  max-width: 100%;  /* 确保内容不会溢出容器 */
}

/* 文章内容样式优化 */
.detail-content h1,
.detail-content h2,
.detail-content h3,
.detail-content h4 {
  margin-top: 2rem;
  margin-bottom: 1rem;
  color: #2c3e50;
  font-weight: 600;
  line-height: 1.4;
}

.detail-content h1 {
  font-size: 1.875rem;
  border-bottom: 2px solid #eef2f7;
  padding-bottom: 0.5rem;
}

.detail-content h2 {
  font-size: 1.5rem;
}

.detail-content h3 {
  font-size: 1.25rem;
}

.detail-content p {
  margin: 1.25rem 0;
  line-height: 1.8;
  color: #4a5568;
}

.detail-content a {
  color: #42b883;
  text-decoration: none;
  border-bottom: 1px solid transparent;
  transition: border-color 0.2s ease;
}

.detail-content a:hover {
  border-bottom-color: #42b883;
}

.detail-content blockquote {
  margin: 1.5rem 0;
  padding: 1rem 1.5rem;
  border-left: 4px solid #42b883;
  background: #f8fafc;
  border-radius: 4px;
  color: #4a5568;
  font-style: italic;
}

.detail-content pre {
  margin: 1.5rem 0;
  padding: 1.25rem;
  border-radius: 8px;
  background: #2d3748 !important;
  overflow-x: auto;
}

/* 调整图片最大宽度 */
.detail-content img {
  max-width: 80%; 
  height: auto;
  display: block;
  margin: 2rem auto;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.code-block-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 1rem;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
}

.code-block-language {
  color: #e2e8f0;
  font-size: 0.875rem;
  font-family: 'Fira Code', monospace;
}

.code-block-actions {
  display: flex;
  gap: 0.5rem;
}

.code-collapse-btn,
.code-copy-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.25rem;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 4px;
  color: #fff;
  cursor: pointer;
  transition: all 0.2s ease;
}

.code-collapse-btn:hover,
.code-copy-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.code-block-content {
  transition: max-height 0.3s ease-in-out;
  max-height: 1000px; /* 调整此值以适应较长的代码块 */
  overflow: hidden;
}

.code-block-content.collapsed {
  max-height: 0;
}

.collapse-icon {
  transition: transform 0.3s ease;
}

.code-block-wrapper {
  margin: 1.5rem 0;
  border-radius: 8px;
  overflow: hidden;
  background: #1a1a1a;
}

/* 修改原有的复制按钮样式 */
.code-copy-btn {
  position: static; /* 取消绝对定位 */
  opacity: 1; /* 始终显示 */
}

/* 移除原有的悬停显示复制按钮的效果 */
.code-block-wrapper:hover .code-copy-btn {
  opacity: 1;
}

/* 代码样式优化 */
.detail-content pre code {
  font-family: 'Fira Code', Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
  font-size: 0.9em;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* 内联代码样式 */
.detail-content :not(pre) > code {
  background: rgba(66, 184, 131, 0.1) !important;
  color: #42b883 !important;
  padding: 0.2em 0.4em !important;
  border-radius: 4px !important;
  font-size: 0.875em !important;
  font-family: 'Fira Code', monospace !important;
}

/* Prism.js 主题覆盖 */
:not(pre) > code[class*="language-"],
pre[class*="language-"] {
  background: #1a1a1a !important;
}

.line-numbers .line-numbers-rows {
  border-right: 1px solid #404040 !important;
  padding-right: 0.5rem;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .detail-container {
    grid-template-columns: 1fr;
    max-width: 900px;
  }

  .detail-toc {
    display: none;
  }
}

/* 优化移动端显示 */
@media (max-width: 768px) {
  .detail-post {
    padding: 0.5rem;
  }
  
  .detail-content {
    padding: 1.5rem;
  }
}
/* 代码高亮样式微调 */
.detail-content :not(pre) > code {
  background: #f1f5f9;
  color: #ef4444;
  padding: 0.2em 0.4em;
  border-radius: 4px;
  font-size: 0.875em;
  font-family: 'Fira Code', monospace;
}

/* 数学公式块样式 */
.detail-content .math-block {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 8px;
  overflow-x: auto;
}
.code-block-wrapper {
  position: relative;
  margin: 1.5rem 0;
  border-radius: 8px;
  overflow: hidden;
  background: #1a1a1a;
}

.code-block-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.75rem 1rem;
  background: #2d2d2d;
  border-bottom: 1px solid #404040;
}

.code-block-language {
  color: #e2e8f0;
  font-size: 0.875rem;
  font-family: monospace;
  text-transform: uppercase;
}

.code-block-actions {
  display: flex;
  gap: 0.5rem;
}

.code-collapse-btn,
.code-copy-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  padding: 0;
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 4px;
  color: #fff;
  cursor: pointer;
  transition: all 0.2s ease;
}

.code-collapse-btn:hover,
.code-copy-btn:hover {
  background: rgba(255, 255, 255, 0.2);
}

.code-block-content {
  transition: max-height 0.3s ease-in-out;
  max-height: 1000px;
  overflow: hidden;
}

.code-block-content.collapsed {
  display: none; /* 折叠时隐藏内容 */
}

.collapse-icon {
  transition: transform 0.3s ease;
}

.collapse-icon.rotate-180 {
  transform: rotate(180deg);
}

.copy-icon,
.check-icon {
  width: 16px;
  height: 16px;
}

.hidden {
  display: none;
}
/* 增加新的样式 */
.detail-container {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
}

.detail-main {
  min-width: 0; /* 防止内容溢出 */
}

.detail-sidebar {
  position: sticky;
  top: 2rem;
  height: calc(100vh - 4rem);
  overflow-y: auto;
  padding-right: 1rem;
  width: 280px;  /* 明确设置宽度 */
}

.sidebar-section {
  background: #ffffff;
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  border: 1px solid #eef2f7;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.sidebar-title {
  font-size: 1rem;
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* 目录样式 */
.toc-nav {
  font-size: 0.875rem;
}

.toc-list, .toc-sublist {
  list-style: none;
  padding-left: 0;
}

.toc-sublist {
  padding-left: 1rem;
}

.toc-link {
  color: #4a5568;
  text-decoration: none;
  display: block;
  padding: 0.25rem 0;
  transition: all 0.2s ease;
  border-left: 2px solid transparent;
  padding-left: 0.5rem;
}

.toc-link:hover {
  color: #42b883;
  border-left-color: #42b883;
}

/* 阅读进度条样式 */
.progress-bar {
  height: 6px;
  background: #edf2f7;
  border-radius: 3px;
  overflow: hidden;
  margin: 0.5rem 0;
}

.progress-inner {
  height: 100%;
  background: linear-gradient(90deg, #42b883, #3aa876);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.875rem;
  color: #718096;
  text-align: center;
}

/* 最近文章列表样式 */
.recent-posts-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.recent-posts-list li {
  margin-bottom: 0.75rem;
}

.recent-posts-list a {
  color: #4a5568;
  text-decoration: none;
  display: block;
  padding: 0.5rem;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.recent-posts-list a:hover {
  background: #f7fafc;
  color: #42b883;
}

.recent-posts-list .active a {
  background: rgba(66, 184, 131, 0.1);
  color: #42b883;
}

.post-date {
  font-size: 0.75rem;
  color: #718096;
  display: block;
  margin-top: 0.25rem;
}

/* 元信息样式优化 */
.detail-meta {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
  margin-top: 1rem;
}

.meta-divider {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #cbd5e0;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .detail-container {
    grid-template-columns: 1fr;
  }
  
  .detail-sidebar {
    display: none;
  }
}
</style>