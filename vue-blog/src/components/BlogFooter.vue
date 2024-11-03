<!-- src/components/BlogFooter.vue -->
<template>
    <footer class="footer">
      <p>© {{ currentYear }} 我的博客. All rights reserved.</p>
      <div class="social-links">
        <a href="#">微博</a> |
        <a href="#">GitHub</a> |
        <a href="#">LinkedIn</a>
      </div>
    </footer>
  </template>
  
  <script>
  export default {
    name: 'BlogFooter',
    computed: {
      currentYear() {
        return new Date().getFullYear()
      }
    }
  }
  </script>