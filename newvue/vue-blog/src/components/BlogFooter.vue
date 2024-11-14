<!-- src/components/BlogFooter.vue -->
<template>
    <footer class="footer">
      <p>© {{ currentYear }} 星云茶聚. All rights reserved.</p>
      <div class="social-links">
        <a href="#">微博</a> |
        <a href="https://github.com/SUXUING-star/SUXUING-star.github.io">GitHub</a> |
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