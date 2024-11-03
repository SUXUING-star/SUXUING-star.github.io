// src/main.js
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import './assets/styles/main.css'

// 导入 KaTeX CSS
import 'katex/dist/katex.min.css'
// 导入代码高亮样式
import 'highlight.js/styles/github.css'



const app = createApp(App)

app.use(store)
app.use(router)

app.mount('#app')