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




