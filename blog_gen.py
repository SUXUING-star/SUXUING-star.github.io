# -*- coding: utf-8 -*-
import os
import sys
import shutil
from datetime import datetime
import markdown
import yaml
from pathlib import Path
import re

class BlogGenerator:
    def __init__(self):
        # 基础目录结构
        self.base_dir = Path('.')
        self.posts_dir = self.base_dir / 'posts'
        self.public_dir = self.base_dir / 'public'
        self.templates_dir = self.base_dir / 'templates'
        
        # 创建必要的目录
        self.posts_dir.mkdir(exist_ok=True)
        self.public_dir.mkdir(exist_ok=True)
        self.templates_dir.mkdir(exist_ok=True)
        (self.public_dir / 'images').mkdir(exist_ok=True)

    def create_new_post(self, title):
        """创建新的博文文件夹和模板"""
        folder_name = title.lower().replace(' ', '-')
        post_dir = self.posts_dir / folder_name
        post_dir.mkdir(exist_ok=True)
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        md_content = f"""---
title: {title}
date: {date_str}
---

在这里写入你的博客内容...
"""
        try:
            with open(post_dir / 'post.md', 'w', encoding='utf-8') as f:
                f.write(md_content)
            (post_dir / 'images').mkdir(exist_ok=True)
            print(f"已创建新博文目录: {post_dir}")
            print(f"请在 {post_dir}/post.md 中编写文章")
            print(f"将图片放在 {post_dir}/images 目录中")
        except Exception as e:
            print(f"创建新文章时出错: {e}")

    def process_post(self, post_dir):
        """处理单个博文文件夹"""
        md_file = post_dir / 'post.md'
        if not md_file.exists():
            return None

        metadata = {
            'title': post_dir.name,
            'date': datetime.now().strftime('%Y-%m-%d')
        }

        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if content.startswith('---'):
                try:
                    _, front_matter, content = content.split('---', 2)
                    yaml_data = yaml.safe_load(front_matter)
                    if yaml_data:
                        metadata.update(yaml_data)
                except Exception as e:
                    print(f"处理文章 {post_dir.name} 的 front matter 时出错: {e}")

        except Exception as e:
            print(f"读取文件 {md_file} 时出错: {e}")
            return None

        images_dir = post_dir / 'images'
        if images_dir.exists():
            post_public_images = self.public_dir / 'images' / post_dir.name
            post_public_images.mkdir(exist_ok=True)
            
            for img in images_dir.glob('*.*'):
                shutil.copy(img, post_public_images / img.name)
                content = content.replace(
                    f'![](images/{img.name})',
                    f'![](/images/{post_dir.name}/{img.name})'
                )

        html_content = markdown.markdown(
            content,
            extensions=['extra', 'codehilite', 'tables', 'toc']
        )

        try:
            template_path = self.templates_dir / 'post_template.html'
            if not template_path.exists():
                print("模板文件 post_template.html 不存在，确保在 templates 目录中存在该文件。")
                return None

            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()

            html = template.format(
                title=metadata.get('title', '无标题'),
                date=metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                content=html_content
            )

            output_file = self.public_dir / f'{post_dir.name}.html'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html)

            return {
                'title': metadata.get('title', '无标题'),
                'date': metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                'url': f'{post_dir.name}.html'
            }

        except Exception as e:
            print(f"生成文章 {post_dir.name} 的 HTML 时出错: {e}")
            return None

    def build(self, original_index='index.html'):
        """构建整个博客"""
        try:
            # 复制 index.html 文件
            shutil.copy(original_index, self.public_dir / 'index.html')
        except Exception as e:
            print(f"复制 index.html 文件时出错: {e}")
            return

        if not (self.templates_dir / 'post_template.html').exists():
            with open(original_index, 'r', encoding='utf-8') as f:
                template = f.read()
            post_template = '''<article class="blog-post">
    <h2 class="post-title">{title}</h2>
    <div class="post-date">{date}</div>
    <div class="post-content">
        {content}
    </div>
</article>'''
            main_start = template.find('<main>') + len('<main>')
            main_end = template.find('</main>')
            template = (
                template[:main_start] +
                post_template +
                template[main_start:main_end] +
                template[main_end:]
            )
            with open(self.templates_dir / 'post_template.html', 'w', encoding='utf-8') as f:
                f.write(template)

        posts_html = []
        for post_dir in self.posts_dir.iterdir():
            if post_dir.is_dir():
                post_info = self.process_post(post_dir)
                if post_info:
                    posts_html.append(f"""
<article class="blog-post">
    <h2 class="post-title"><a href="{post_info['url']}">{post_info['title']}</a></h2>
    <div class="post-date">{post_info['date']}</div>
</article>
                    """)

        try:
            # 读取 post_template.html
            with open(self.templates_dir / 'post_template.html', 'r', encoding='utf-8') as f:
                index_content = f.read()

            main_content = '\n'.join(posts_html)
            start_tag = '<main>'
            end_tag = '</main>'
            start_pos = index_content.find(start_tag) + len(start_tag)
            end_pos = index_content.find(end_tag)

            new_index_content = (
                index_content[:start_pos] +
                main_content +
                index_content[end_pos:]
            )

            with open(self.public_dir / 'index.html', 'w', encoding='utf-8') as f:
                f.write(new_index_content)
        except Exception as e:
            print(f"更新首页时出错: {e}")

if __name__ == "__main__":
    generator = BlogGenerator()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'new':
            if len(sys.argv) > 2:
                generator.create_new_post(' '.join(sys.argv[2:]))
            else:
                print("请提供文章标题！")
        elif sys.argv[1] == 'build':
            generator.build('index.html')
        else:
            print("未知命令！使用 'new' 创建新文章或 'build' 构建网站")
    else:
        print("""
使用方法：
1. 创建新文章：py -3.11 blog_gen.py new "文章标题"
2. 构建网站：py -3.11 blog_gen.py build
        """)
