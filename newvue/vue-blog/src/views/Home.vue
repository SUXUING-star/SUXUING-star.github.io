<!-- Home.vue -->
<template>
  <div class="home-page">
    <section class="banner">
      <div class="banner-content">
        <h2>欢迎来到这个神奇的地方！</h2>
        <p>记录成长，探索未知的世界。</p>
      </div>
    </section>
    
    <main class="posts-grid">
      <blog-post
        v-for="post in paginatedPosts"
        :key="post.id"
        :post="post"
        @click="viewPost(post.id)"
      />
    </main>

    <!-- 分页控件 -->
    <div v-if="totalPages > 1" class="pagination">
      <button 
        class="pagination-btn" 
        :disabled="currentPage === 1"
        @click="changePage(currentPage - 1)"
      >
        ← 上一页
      </button>
      
      <div class="pagination-numbers">
        <button 
          v-for="page in displayedPages" 
          :key="page"
          class="page-number"
          :class="{ active: page === currentPage }"
          @click="changePage(page)"
        >
          {{ page }}
        </button>
      </div>

      <button 
        class="pagination-btn"
        :disabled="currentPage === totalPages"
        @click="changePage(currentPage + 1)"
      >
        下一页 →
      </button>
    </div>
  </div>
</template>
  
<script>
import { useStore } from 'vuex'
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import BlogPost from '@/components/BlogPost.vue'

export default {
  name: 'HomePage',
  components: {
    BlogPost
  },
  setup() {
    const store = useStore()
    const router = useRouter()
    
    // 分页相关的响应式数据
    const postsPerPage = 6  // 每页显示的文章数
    const currentPage = ref(1)
    
    const posts = computed(() => store.state.posts)
    
    // 计算总页数
    const totalPages = computed(() => Math.ceil(posts.value.length / postsPerPage))
    
    // 当前页的文章
    const paginatedPosts = computed(() => {
      const start = (currentPage.value - 1) * postsPerPage
      const end = start + postsPerPage
      return posts.value.slice(start, end)
    })
    
    // 计算要显示的页码（最多显示5个页码）
    const displayedPages = computed(() => {
      const total = totalPages.value
      const current = currentPage.value
      const pages = []
      
      if (total <= 5) {
        // 如果总页数小于等于5，显示所有页码
        for (let i = 1; i <= total; i++) {
          pages.push(i)
        }
      } else {
        // 如果总页数大于5，显示当前页附近的页码
        if (current <= 3) {
          // 当前页靠近开始
          for (let i = 1; i <= 5; i++) {
            pages.push(i)
          }
        } else if (current >= total - 2) {
          // 当前页靠近结束
          for (let i = total - 4; i <= total; i++) {
            pages.push(i)
          }
        } else {
          // 当前页在中间
          for (let i = current - 2; i <= current + 2; i++) {
            pages.push(i)
          }
        }
      }
      
      return pages
    })
    
    // 切换页面
    const changePage = (page) => {
      currentPage.value = page
      // 滚动到页面顶部
      window.scrollTo({
        top: 0,
        behavior: 'smooth'
      })
    }
    
    const viewPost = (id) => {
      router.push(`/post/${id}`)
    }
    
    return {
      posts,
      paginatedPosts,
      currentPage,
      totalPages,
      displayedPages,
      changePage,
      viewPost
    }
  }
}
</script>

<style scoped>
/* 保留原有样式 */
.home-page {
  max-width: 1600px;
  margin: 0 auto;
  padding: 1rem;
}


.posts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  padding: 0.5rem 0;
  max-width: 1000px;
  margin: 0 auto;
}

/* 新增分页相关样式 */
.pagination {
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 2rem 0;
  gap: 1rem;
}

.pagination-numbers {
  display: flex;
  gap: 0.5rem;
}

.pagination-btn, .page-number {
  padding: 0.5rem 1rem;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  background-color: #ffffff;
  color: #4a5568;
  cursor: pointer;
  transition: all 0.2s ease;
}

.pagination-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.pagination-btn:not(:disabled):hover,
.page-number:hover {
  background-color: #f7fafc;
  border-color: #cbd5e0;
}

.page-number.active {
  background-color: #42b883;
  color: white;
  border-color: #42b883;
}

@media (max-width: 768px) {
  .banner {
    padding: 1.5rem 1rem;
  }

  .banner h2 {
    font-size: 1.5rem;
  }

  .banner p {
    font-size: 1rem;
  }

  .posts-grid {
    grid-template-columns: 1fr;
    gap: 1.25rem;
  }

  .pagination {
    flex-wrap: wrap;
  }
}
</style>