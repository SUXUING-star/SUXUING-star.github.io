<template>
  <article class="blog-post">
    <h2 class="post-title">
      <a href="#" @click.prevent="$emit('click')">{{ post.title }}</a>
    </h2>
    <div class="post-date">{{ post.date }}</div>
    <div class="post-content">
      <p>{{ post.summary }}</p>
    </div>
    <div v-if="post.coverImage" class="cover-image">
      <img 
        :src="post.coverImage" 
        :alt="post.title"
        @error="handleImageError"
      >
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
  cursor: pointer;
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
.cover-image img{
  max-width: 70%;        /* 最大宽度为父容器的一半 */
  height: auto;          /* 保持纵横比 */
  display: block;        /* 确保图片是块级元素 */
  margin-left: auto;     /* 左右居中 */
  margin-right: auto;    /* 左右居中 */
  
  border: 5px solid #ccc;    /* 给图片添加边框 */
  border-radius: 10px;        /* 圆角效果 */
  padding: 10px;              /* 图片四周加内边距（留白） */
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* 添加阴影效果 */
}
</style>