---
title: 一个简单的文本情感分析模型
date: 2024-11-01
summary: 这是全部的代码的描述和一些注释
coverImage: post2/post2-1.png
---
## 自己简单爬取的数据进行分析
```javascript
// ==UserScript==
// @name         amazon-scraper1
// @namespace    http://tampermonkey.net/
// @version      2024-11-06
// @description  try to take over the world!
// @author       suxing
// @match        https://www.amazon.com/*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=amazon.com
// @grant        none
// ==/UserScript==



function main(){
    new Promise((resolve)=>{
        console.log("amazonbutton");
        setTimeout(()=>{
            resolve();
        },2000)
    }).then(()=>{
        let pageMain = document.querySelector(".sg-col-inner");
        console.log(pageMain)
        if(pageMain){
            let button = document.createElement("button");
            button.className="button-test"
            button.innerHTML = "点击爬取json";
            button.style.padding = "10px 20px";
            button.style.fontSize = "16px";
            button.style.backgroundColor = "#4CAF50";
            button.style.color = "white";
            button.style.border = "none";
            button.style.borderRadius = "5px";
            button.style.cursor = "pointer";
            button.style.boxShadow = "0px 4px 6px rgba(0, 0, 0, 0.1)";
            button.style.transition = "background-color 0.3s";
            button.onmouseover = function() {
                button.style.backgroundColor = "#45a049";
            };
            button.onmouseout = function() {
                button.style.backgroundColor = "#4CAF50";
            };
            button.onclick=()=>{scrapefunc();}
            var buttonContainer = document.createElement("div");
            buttonContainer.style.display = "flex";
            buttonContainer.style.justifyContent = "center";
            buttonContainer.style.alignItems = "center";
            buttonContainer.style.height = "100px"; // 调整高度以便更好地居中
            buttonContainer.appendChild(button);

            // 在 pageMain 元素的上方插入按钮
            pageMain.parentNode.insertBefore(buttonContainer, pageMain);
        }
    })
}

function exportjson(data){
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const today = new Date();
    const year = today.getFullYear();
    const month = String(today.getMonth() + 1).padStart(2, '0');
    const day = String(today.getDate()).padStart(2, '0');
    const formattedDate = `${year}-${month}-${day}`;
    a.download = `amazon-Products${formattedDate}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
function scrapefunc(){
    new Promise((resolve)=>{
        console.log("amazon-scraper test1");
        setTimeout(()=>{
            resolve();
        },700)
    }).then(()=>{
        new Promise((resolve)=>{
            window.scrollTo({
                top: 1000,
                behavior: "smooth"
            });
            setTimeout(()=>{
                resolve();
            },1000)
        }).then(()=>{
            new Promise((resolve)=>{
                window.scrollTo({
                    top: 4000,
                    behavior: "smooth"
                });
                setTimeout(()=>{
                    resolve();
                },1000)
            }).then(()=>{
                new Promise((resolve)=>{
                    window.scrollTo({
                        top: 8000,
                        behavior: "smooth"
                    });
                    setTimeout(()=>{
                        resolve();
                    },1000)
                }).then(()=>{
                    new Promise((resolve)=>{
                        window.scrollTo({
                            top: document.body.scrollHeight,
                            behavior: "smooth"
                        });
                        setTimeout(()=>{
                            resolve();
                        },700)
                    }).then(()=>{
                        let productlist = [];
                        document.querySelectorAll('div.a-section.a-spacing-small.puis-padding-left-small.puis-padding-right-small').forEach(container => {
                            let product = {};

                            // 标题和URL: 从商品标题容器获取
                            try {
                                const titleContainer = container.querySelector('[data-cy="title-recipe"]');
                                product.title = titleContainer.querySelector('.a-size-base-plus.a-color-base').textContent.trim();
                                product.url = titleContainer.querySelector('a.a-link-normal').getAttribute('href');
                            } catch (e) {
                                product.title = '';
                                product.url = '';
                            }

                            // 价格: 从价格容器获取
                            try {
                                const priceContainer = container.querySelector('[data-cy="price-recipe"]');
                                const priceElement = priceContainer.querySelector('.a-price .a-offscreen');
                                product.price = priceElement ? priceElement.textContent.trim() : '';
                            } catch (e) {
                                product.price = '';
                            }

                            // 评分和评分数: 从评论容器获取
                            try {
                                const reviewsContainer = container.querySelector('[data-cy="reviews-block"]');
                                const ratingElement = reviewsContainer.querySelector('.a-icon-alt');
                                product.rating = ratingElement ? ratingElement.textContent.trim() : '';

                                const reviewsElement = reviewsContainer.querySelector('.rush-component .s-underline-text');
                                product.ratingnum = reviewsElement ? reviewsElement.textContent.trim() : '';
                            } catch (e) {
                                product.rating = '';
                                product.ratingnum = '';
                            }

                            if (product.title) {  // 只添加有标题的商品
                                productlist.push(product);
                            }
                        });
                        const productListJson = JSON.stringify(productlist, null, 2);
                        console.log(productListJson);
                        exportjson(productListJson);
                        const nexturl=document.querySelector(".s-pagination-next").getAttribute("href")
                        window.scrollTo({
                            top: 0,
                            behavior: "smooth"
                        });
                        new Promise((resolve)=>{
                            setTimeout(()=>{
                                resolve();
                            },500)
                        }).then(()=>{
                            location.href=nexturl
                        })

                    })
                })
            })
        })
    })
}





window.addEventListener("load",()=>{
    main();

},false)
```


## 深度学习模型构建部分

### 简要数据分析
```python
# 导入所需库
from datasets import load_dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

# 加载数据集，并取前2000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:10000])  # 仅取2000条数据
print(data.head())

# 1. 词云生成
all_text = " ".join(data["text"].fillna(""))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Amazon Reviews")
plt.show()

# 2. 用户画像分析
# 评分分布
plt.figure(figsize=(8, 5))
sns.histplot(data["rating"], bins=5, kde=True)
plt.xlabel("Rating")
plt.title("Distribution of Ratings")
plt.show()

# 验证购买分析
verified_purchase_counts = data["verified_purchase"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=verified_purchase_counts.index, y=verified_purchase_counts.values)
plt.xlabel("Verified Purchase")
plt.ylabel("Count")
plt.title("Distribution of Verified Purchases")
plt.show()

# 3. 时间序列分析：评论随时间的变化
data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")  # 转换时间戳为日期格式
data.set_index("timestamp", inplace=True)
data["rating"].resample("M").mean().plot(figsize=(12, 6))
plt.title("Average Rating Over Time (Monthly)")
plt.xlabel("Time")
plt.xlim(pd.Timestamp("2012-01-01"), pd.Timestamp("2023-12-31"))
plt.ylabel("Average Rating")
plt.show()

# 4. 评论字数分析
data["review_length"] = data["text"].apply(lambda x: len(str(x).split()))  # 计算每条评论的词数
data["review_length"].resample("M").mean().plot(figsize=(12, 6))
plt.title("Average Review Length Over Time (Monthly)")
plt.xlabel("Time")
plt.xlim(pd.Timestamp("2012-01-01"), pd.Timestamp("2023-12-31"))
plt.ylabel("Average Review Length")
plt.show()





```