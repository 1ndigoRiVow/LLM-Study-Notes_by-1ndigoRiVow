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
        text: '学习笔记',
        items: [
          { text: '矩阵分解', link: '/guide/learning/p1' },
          { text: 'Token', link: '/guide/learning/token' },
          { text: 'Self-Attention', link: '/guide/learning/attention' }
        ]
      },
      {
        text: '模型实操',
        items: [
          { text: '🛠️（施工中）', link: '/guide/llmexp/' }
        ]
      },
      {
        text: '项目经验',
        items: [
          { text: '🛠️（施工中）', link: '/guide/projexp/' }
        ]
      },
      {
        text: '经济学研究',
        items: [
          { text: '总览', link: '/guide/econ/p1' },
          { text: 'IV', link:'/guide/econ/iv' },
          { text: 'GMM', link:'/guide/econ/gmm' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
