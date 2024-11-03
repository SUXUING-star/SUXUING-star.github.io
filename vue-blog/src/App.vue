<!-- src/App.vue -->
<template>
  <div class="app">
    <blog-header />
    <router-view />
    <blog-footer />
  </div>
</template>

<script>
import BlogHeader from '@/components/BlogHeader.vue'
import BlogFooter from '@/components/BlogFooter.vue'

export default {
  name: 'App',
  components: {
    BlogHeader,
    BlogFooter
  }
}
</script>
