<template>
    <div class="post-detail" v-if="post">
      <div class="post-container">
        <h1>{{ post.title }}</h1>
        <div class="post-meta">
          <span class="date">{{ post.date }}</span>
        </div>
        <div class="post-content markdown-body" v-html="renderedContent" ref="contentRef"></div>
      </div>
    </div>
  </template>
  
  <script>
  import { computed, onMounted, ref, watch, nextTick } from 'vue'
  import { useRoute } from 'vue-router'
  import { useStore } from 'vuex'
  import { marked } from 'marked'
  
  // 正确的 Prism.js 导入顺序
  import Prism from 'prismjs'
  import 'prismjs/themes/prism-tomorrow.css'
  // 在核心加载后再加载语言
  import 'prismjs/components/prism-markup'
  import 'prismjs/components/prism-css'
  import 'prismjs/components/prism-javascript'
  import 'prismjs/components/prism-typescript'
  import 'prismjs/components/prism-python'
  import 'prismjs/components/prism-bash'
  import 'prismjs/components/prism-json'
  // 可选的插件
  import 'prismjs/plugins/line-numbers/prism-line-numbers'
  import 'prismjs/plugins/line-numbers/prism-line-numbers.css'
  
  export default {
    name: 'PostDetailPage',
    setup() {
      const route = useRoute()
      const store = useStore()
      const contentRef = ref(null)
  
      const loadMathJax = () => {
        return new Promise((resolve) => {
          window.MathJax = {
            tex: {
              inlineMath: [['$', '$']],
              displayMath: [['$$', '$$']]
            },
            svg: {
              fontCache: 'global'
            },
            startup: {
              pageReady: () => {
                resolve()
              }
            }
          }
  
          const script = document.createElement('script')
          script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js'
          document.head.appendChild(script)
        })
      }
      const getImageUrl = (imagePath) => {
        if (!imagePath) return ''
        if (imagePath.startsWith('http')) {
          return imagePath
        }
        try {
          console.warn(`Trying to load image from: @/posts/images/${imagePath}`) // 输出尝试加载的图片路径
          return require(`@/posts/images/${imagePath}`)
        } catch (e) {
          console.warn(`Image not found: ${imagePath}`)
          return ''
        }
      }
      const configureMarked = () => {
        const renderer = new marked.Renderer()
  
        // 优化的代码高亮处理
        renderer.code = (code, language = 'plaintext') => {
          let highlightedCode = code
          try {
            if (Prism.languages[language]) {
              highlightedCode = Prism.highlight(
                code,
                Prism.languages[language],
                language
              )
            }
  
            if (language === 'math') {
              return `<div class="math-block">$$${code}$$</div>`
            }
  
            return `
              <div class="code-block-wrapper">
                <pre class="line-numbers language-${language}"><code>${highlightedCode}</code></pre>
              </div>
            `
          } catch (err) {
            console.error('Code highlighting error:', err)
            return `<pre><code>${code}</code></pre>`
          }
        }
        // 修改图片渲染逻辑
        renderer.image = (href, title, text) => {
          const imageUrl = getImageUrl(href)
          return `
            <div class="image-container">
              <img src="${imageUrl}" alt="${text}" title="${title || ''}" loading="lazy" />
              ${text ? `<div class="image-caption">${text}</div>` : ''}
            </div>
          `
        }
  
        marked.setOptions({
          renderer,
          gfm: true,
          breaks: true,
          pedantic: false,
          smartLists: true,
          smartypants: false
        })
      }
  
      const renderMathJax = () => {
        if (window.MathJax) {
          window.MathJax.typeset()
        }
      }
  
      const highlightCode = () => {
        if (contentRef.value) {
          const codeBlocks = contentRef.value.querySelectorAll('pre code')
          codeBlocks.forEach((block) => {
            Prism.highlightElement(block)
          })
        }
      }
     onMounted(async () => {
        configureMarked()
        await loadMathJax()
        renderMathJax() // 首次加载后渲染一次公式
      })

      const post = computed(() => 
        store.state.posts.find(p => p.id === parseInt(route.params.id))
      )
      const processMarkdownContent = (content) => {
        if (!content) return ''
        // 处理 Markdown 内容中的图片路径
        return content.replace(
          /!\[(.*?)\]\((.*?)\)/g,
          (match, alt, path) => {
            if (path.startsWith('http')) {
              return match
            }
            const imageUrl = getImageUrl(path)
            return `![${alt}](${imageUrl})`
          }
        )
      }
      const renderedContent = computed(() => {
        if (!post.value?.content) return ''
        try {
          // 先处理内容中的图片路径，再进行 Markdown 渲染
          const processedContent = processMarkdownContent(post.value.content)
          return marked.parse(processedContent)
        } catch (err) {
          console.error('Markdown rendering error:', err)
          return '<p>Error rendering content</p>'
        }
      })
    
      watch(renderedContent, () => {
        nextTick(() => {
          highlightCode()
          renderMathJax()
        })
      })
      
      return {
        post,
        renderedContent,
        contentRef
      }
    }
  }
  </script>
  
 <style>
 
  /* Prism.js 基础样式覆盖 */
  :root {
    --prism-background: #2d2d2d;
    --prism-color: #e0e0e0;
    --mathjax-bg: #f3f4f6;
    --post-bg: #f7fbfc;
    --post-color: #333;
  }
  
  .code-block-wrapper {
    margin: 1.5rem 0;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  pre[class*="language-"] {
    margin: 0;
    padding: 1rem;
    background: var(--prism-background) !important;
    color: var(--prism-color);
    font-size: 0.9rem;
    line-height: 1.6;
    tab-size: 2;
    border-radius: 8px;
  }
  
  code[class*="language-"] {
    font-family: 'Fira Code', Consolas, Monaco, monospace;
    white-space: pre-wrap;
    color: var(--prism-color);
  }
  
  /* 行号样式 */
  .line-numbers .line-numbers-rows {
    border-right: 1px solid #555;
    padding-right: 0.75rem;
  }

  /* MathJax 样式 */
  .math-block {
    overflow-x: auto;
    padding: 1.2rem;
    margin: 1.5rem 0;
    background: var(--mathjax-bg);
    border-radius: 8px;
    color: #333;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  }

  .MathJax {
    font-size: 1.05em;
    color: #2c3e50;
  }
  
  /* 文章容器样式 */
  .post-container {
    max-width: 800px;
    margin: 0 auto;
    background-color: var(--post-bg);
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    border: 1px solid #e0e0e0;
  }
  
  /* 文章内容样式 */
  .post-content {
    line-height: 1.8;
    color: var(--post-color);
  }
  
  .post-content h1,
  .post-content h2,
  .post-content h3 {
    margin-top: 1.8rem;
    margin-bottom: 1.2rem;
    font-weight: 600;
    color: #2c3e50;
  }
  
  .post-content p {
    margin: 1rem 0;
  }
  
  .post-content blockquote {
    border-left: 4px solid #42b883;
    margin: 1.5rem 0;
    padding: 0.8rem 1.2rem;
    background-color: #e0f5f1;
    color: #555;
    font-style: italic;
    border-radius: 5px;
  }
  
  /* 行内代码样式 */
  :not(pre) > code {
    background: #f5f5f5;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-size: 0.9em;
    color: #d63384;
  }
</style>
