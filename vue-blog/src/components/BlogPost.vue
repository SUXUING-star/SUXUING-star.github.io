<template>
  <article class="blog-post">
    <h2 class="post-title">
      <a href="#" @click.prevent="$emit('click')">{{ post.title }}</a>
    </h2>
    <div class="post-date">{{ post.date }}</div>

    <div v-if="post.coverImage" class="cover-image">
      <img 
        :src="post.coverImage" 
        :alt="post.title"
        @error="handleImageError"
      >
    </div>

    <div class="post-content">
      <p>{{ post.summary }}</p>
    </div>
  </article>
</template>

<script>
export default {
  name: 'BlogPostItem',
  props: {
    post: {
      type: Object,
      required: true
    }
  },
  methods: {
    getImageUrl(imagePath) {
      try {
        return require(`@/posts/images/${imagePath}`); // 确保路径正确
      } catch (e) {
        console.warn(`Image not found: ${imagePath}`);
        return '';
      }
    }
  }
}
</script>

  
  
  
  <style scoped>
.blog-post {
  font-family: Arial, sans-serif;
  padding: 1rem;
  background-color: #f9f9f9; /* 主背景色 */
  border-radius: 8px;
}

.post-content {
  line-height: 1.6;
  color: #333;
}

/* 设置代码块样式 */
.post-content pre {
  background-color: #2e3440; /* 代码块背景色 */
  color: #d8dee9; /* 代码块字体颜色 */
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  font-family: Consolas, monospace;
  margin: 1.5rem 0;
}

/* 代码块中的代码 */
.post-content code {
  background-color: #3b4252;
  color: #d8dee9;
  padding: 0.2rem 0.4rem;
  border-radius: 4px;
}

/* 行内代码 */
.post-content p code {
  background-color: #eceff4;
  color: #bf616a;
  padding: 0.2rem 0.3rem;
  font-size: 0.9rem;
  border-radius: 3px;
}
</style>