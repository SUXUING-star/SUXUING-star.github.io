<!-- src/views/Archive.vue -->
<template>
    <div class="archive">
      <h2>文章归档</h2>
      <ul class="archive-list">
        <li v-for="post in posts" :key="post.id">
          <span class="post-date">{{ post.date }}</span>
          <a href="#" @click.prevent="viewPost(post.id)">{{ post.title }}</a>
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  import { useStore } from 'vuex'
  import { computed } from 'vue'
  import { useRouter } from 'vue-router'
  
  export default {
    name: 'ArchivePage',
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