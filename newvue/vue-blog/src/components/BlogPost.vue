<!-- BlogPost.vue -->
<template>
  <article class="blog-post-card" @click="$emit('click')">
    <div class="post-image" >
      <img 
        v-if="post.coverImage"
        :src="post.coverImage" 
        :alt="post.title"
        @error="handleImageError"
        loading="lazy"
      >
      <img 
        v-else 
        src="/placeholder-image.gif" 
        alt="Placeholder Image" 
        loading="lazy"
      >
    </div>
    <div class="post-content">
      <h2 class="post-title">{{ post.title }}</h2>
      <div class="post-meta">
        <span class="post-date">{{ post.date }}</span>
      </div>
      <p class="post-summary">{{ post.summary }}</p>
      <div class="post-footer">
        <span class="read-more">阅读全文 →</span>
      </div>
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
    
    
  }
}
</script>

<style scoped>
.blog-post-card {
  background: #ffffff;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  height: 100%;
  display: flex;
  flex-direction: column;
  cursor: pointer;
  border: 1px solid #eef2f7;
}

.blog-post-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.post-image {
  width: 100%;
  height: 180px;
  overflow: hidden;
}

.post-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.blog-post-card:hover .post-image img {
  transform: scale(1.05);
}

.post-content {
  padding: 1.25rem;
  flex: 1;
  display: flex;
  flex-direction: column;
  background: linear-gradient(to bottom, #ffffff 0%, #fafbfc 100%);
}

.post-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #2c3e50;
  margin: 0 0 0.5rem 0;
  line-height: 1.4;
}

.post-meta {
  font-size: 0.875rem;
  color: #718096;
  margin-bottom: 0.75rem;
}

.post-summary {
  color: #4a5568;
  line-height: 1.6;
  margin-bottom: 1rem;
  flex: 1;
  font-size: 0.9375rem;
}

.post-footer {
  margin-top: auto;
  padding-top: 0.75rem;
  border-top: 1px solid #edf2f7;
}

.read-more {
  color: #42b883;
  font-weight: 500;
  font-size: 0.875rem;
}

.read-more:hover {
  color: #3aa876;
}
</style>