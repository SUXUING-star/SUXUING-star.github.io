from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from transformers import BertModel, BertTokenizer
import torch
# 导入所需库
from datasets import load_dataset
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 加载数据集，并取前1000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:1000])  # 仅取1000条数据

# 载入BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# 获取每个用户的评论并进行预处理（取每个用户的所有评论拼接为一个文本）
user_texts = data.groupby("user_id")["text"].apply(lambda x: " ".join(x)).reset_index()

# 使用BERT生成文本嵌入
def get_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        # 获取 [CLS] token 的嵌入表示作为句子表示
        cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
        embeddings.append(cls_embedding[0])
    return embeddings

user_embeddings = get_embeddings(user_texts["text"])

# 计算用户之间的相似度矩阵 (余弦相似度)
cosine_similarity_matrix = cosine_similarity(user_embeddings)

# 计算用户之间的相似度矩阵 (皮尔逊相似度)
# 使用 pairwise_distances 计算距离矩阵，然后转换为相似度矩阵
pearson_distance_matrix = pairwise_distances(user_embeddings, metric="correlation")
pearson_similarity_matrix = 1 - pearson_distance_matrix

# 创建相似度 DataFrame
cosine_similarity_df = pd.DataFrame(cosine_similarity_matrix, index=user_texts["user_id"], columns=user_texts["user_id"])
pearson_similarity_df = pd.DataFrame(pearson_similarity_matrix, index=user_texts["user_id"], columns=user_texts["user_id"])


# 随机选择10个用户用于相似度热力图展示
sampled_users = user_texts.sample(10, random_state=42)
sampled_user_ids = sampled_users["user_id"].values

# 提取这10个用户的相似度矩阵
sampled_cosine_similarity_df = cosine_similarity_df.loc[sampled_user_ids, sampled_user_ids]
sampled_pearson_similarity_df = pearson_similarity_df.loc[sampled_user_ids, sampled_user_ids]

# 可视化相似度热力图 (余弦相似度)
plt.figure(figsize=(10, 8))
sns.heatmap(sampled_cosine_similarity_df, annot=True, cmap="coolwarm", square=True, cbar_kws={'label': 'Cosine Similarity'})
plt.title("User Similarity Matrix (Cosine Similarity)")
plt.show()

# 可视化相似度热力图 (皮尔逊相似度)
plt.figure(figsize=(10, 8))
sns.heatmap(sampled_pearson_similarity_df, annot=True, cmap="coolwarm", square=True, cbar_kws={'label': 'Pearson Similarity'})
plt.title("User Similarity Matrix (Pearson Similarity)")
plt.show()