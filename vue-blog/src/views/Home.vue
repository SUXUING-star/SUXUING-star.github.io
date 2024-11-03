<!-- src/views/Home.vue -->
<template>
    <div class="home-page">
      <section class="banner">
        <h2>欢迎来到我的博客！</h2>
        <p>分享生活，记录成长，探索未知的世界。</p>
      </section>
      
      <main>
        <blog-post
          v-for="post in posts"
          :key="post.id"
          :post="post"
          @click="viewPost(post.id)"
        />
      </main>
    </div>
  </template>
  
  <script>
  import { useStore } from 'vuex'
  import { computed } from 'vue'
  import { useRouter } from 'vue-router'
  import BlogPost from '@/components/BlogPost.vue'
  
  export default {
    name: 'HomePage',  // 改为多词组件名
    components: {
      BlogPost
    },
    setup() {
      const store = useStore()
      const router = useRouter()
      
      const posts = computed(() => store.state.posts)
      
      const viewPost = (id) => {
        router.push(`/post/${id}`)
      }
      
      return {
        posts,
        viewPost
      }
    }
  }
  </script>
  
  