import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({

  base: '/LLM-Study-Notes_by-1ndigoRiVow/', 
  title: "LLM Study Notes",
  description: "其实我还是最喜欢经济学了",
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '首页', link: '/' },
      { text: '开始', link:'/guide/getting-started'},
      { text: 'md示例', link: '/markdown-examples' }
    ],

    sidebar: [
      {
        text: '开始阅读',
        items: [
          { text: '学习笔记', link: '/guide/learning/p1' },
          { text: '模型实操', link: '/guide/llmexp/' },
          { text: '项目经验', link: '/guide/projexp/' },
          { text: '行业趋势', link: '/guide/industry/' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})