import { createRouter, createWebHashHistory } from 'vue-router'
import Home from '../views/Home.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home,
    // 预加载其他常用组件
    beforeEnter: (to, from, next) => {
      const components = [
        import(/* webpackChunkName: "about" */ '../views/About.vue'),
        import(/* webpackChunkName: "archive" */ '../views/Archive.vue')
      ];
      Promise.all(components).catch(() => {});
      next();
    }
  },
  {
    path: '/about',
    name: 'About',
    component: () => import(/* webpackChunkName: "about" */ '../views/About.vue'),
    // 添加 meta 字段用于控制缓存
    meta: { keepAlive: true }
  },
  {
    path: '/post/:id',
    name: 'PostDetail',
    component: () => import(/* webpackChunkName: "post" */ '../views/PostDetail.vue'),
    props: true,
  },
  {
    path: '/archive',
    name: 'Archive',
    component: () => import(/* webpackChunkName: "archive" */ '../views/Archive.vue'),
    meta: { keepAlive: true }
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
  // 滚动行为
  scrollBehavior(to, from, savedPosition) {
    if (savedPosition) {
      return savedPosition;
    } else {
      return { top: 0 };
    }
  }
});

// 路由守卫中添加进度条
router.beforeEach((to, from, next) => {
  // 可以在这里添加 NProgress 等加载进度条
  next();
});

export default router;