<!-- BlogHeader.vue -->
<template>
  <header class="header" :class="{ 'header-scrolled': isScrolled }">
    <div class="header-content">
      <div class="header-main">
        <h1 class="site-title">星云茶聚</h1>
        <nav class="site-nav">
          <router-link to="/" class="nav-link">首页</router-link>
          <router-link to="/archive" class="nav-link">归档</router-link>
          <router-link to="/about" class="nav-link">关于</router-link>
        </nav>
      </div>
    </div>
  </header>
</template>

<script>
export default {
  name: 'BlogHeader',
  data() {
    return {
      isScrolled: false,
      lastScrollTop: 0,
      headerHeight: 0,
      scrollThreshold: 50, // 添加滚动阈值
      scrollTimer: null
    }
  },
  mounted() {
    this.headerHeight = this.$el.offsetHeight
    window.addEventListener('scroll', this.handleScroll, { passive: true })
    document.body.style.paddingTop = `${this.headerHeight + 40}px`
  },
  beforeUnmount() {
    window.removeEventListener('scroll', this.handleScroll)
    document.body.style.paddingTop = '0'
    if (this.scrollTimer) {
      clearTimeout(this.scrollTimer)
    }
  },
  methods: {
    handleScroll() {
      // 使用 requestAnimationFrame 优化性能
      if (this.scrollTimer) {
        cancelAnimationFrame(this.scrollTimer)
      }
      
      this.scrollTimer = requestAnimationFrame(() => {
        const st = window.scrollY // 替换废弃的 pageYOffset
        
        // 在顶部或接近顶部时始终显示
        if (st <= this.scrollThreshold) {
          this.isScrolled = false
          return
        }

        // 计算滚动距离和方向
        const scrollDelta = st - this.lastScrollTop
        
        // 只有当滚动距离超过阈值时才触发变化
        if (Math.abs(scrollDelta) > this.scrollThreshold) {
          if (scrollDelta > 0) {
            // 向下滚动
            this.isScrolled = false
          } else {
            // 向上滚动
            this.isScrolled = true
          }
          this.lastScrollTop = st
        }
      })
    }
  }
}
</script>

<style scoped>

.header-scrolled {
  top: 0;
}

.header {
  background: rgba(255, 255, 255, 0.90); /* 降低不透明度 */
  backdrop-filter: blur(8px); /* 添加模糊效果增加可读性 */
  border-bottom: 1px solid #eef2f7;
  max-width: 900px;
  margin: auto;
  padding: 1.5rem 0;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
}

.header-main {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.05rem;
}

.site-title {
  font-size: 1.5rem;
  font-weight: 700;
  color: #2c3e50;
  margin: 0;
  background: linear-gradient(120deg, #42b883 0%, #3aa876 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.site-nav {
  display: flex;
  gap: 2rem; /* 增加导航项间距 */
}

.nav-link {
  color: #4a5568;
  text-decoration: none;
  font-weight: 500;
  font-size: 0.9375rem;
  padding: 0.5rem 0.75rem; /* 增加可点击区域 */
  position: relative;
  transition: color 0.2s ease;
}
.nav-link:hover {
  color: #42b883;
}

.nav-link::after {
  content: '';
  position: absolute;
  bottom: -2px;
  left: 0;
  width: 100%;
  height: 2px;
  background: #42b883;
  transform: scaleX(0);
  transition: transform 0.2s ease;
}

.nav-link:hover::after,
.router-link-active::after {
  transform: scaleX(1);
}

@media (max-width: 768px) {
  .header-content {
    padding: 0 1rem;
  }
  
  .header-main {
    flex-direction: column;
    gap: 0.5rem;
  }

  .site-nav {
    gap: 1.5rem;
  }
}
</style>