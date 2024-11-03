// src/router/index.js
import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/about',
    name: 'About',
    // 路由懒加载
    component: () => import('../views/About.vue')
  },
  {
    path: '/post/:id',
    name: 'PostDetail',
    component: () => import('../views/PostDetail.vue'),
    props: true
  },
  {
    path: '/archive',
    name: 'Archive',
    component: () => import('../views/Archive.vue')
  }
]

const router = createRouter({
  // 使用 hash 模式，这样部署到 GitHub Pages 不会有路径问题
  history: createWebHashHistory(),
  routes
})

export default router