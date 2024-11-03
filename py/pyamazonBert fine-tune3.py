from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from textaugment import EDA  # 数据增强库
from textblob import TextBlob  # 用于情感分析
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:200000])

# 数据增强函数：对3-4星评论进行同义词替换
augmenter = EDA()
def augment_text(text):
    return augmenter.synonym_replacement(text)

# 文本预处理函数
def preprocess_text(text):
    if pd.isna(text):
        return ""
    return str(text)[:512]

data['text'] = data['text'].apply(preprocess_text)
def plot_confusion_matrix(y_true, y_pred,epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    import os
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 保存图像而不是显示
    plt.savefig("figures/epoch%s-matrix.png"%epoch)
    plt.close()
# 数据标签转换
def convert_rating_to_class(rating):
    if rating <= 2:
        return 0
    elif rating <= 4:
        return 1
    return 2

data['class'] = data['rating'].apply(convert_rating_to_class)

# 情感得分和长度特征添加
def add_sentiment_length_features(df):
    sentiment_scores, text_lengths = [], []
    for text in df['text']:
        blob = TextBlob(text)
        sentiment_scores.append(blob.sentiment.polarity)  # 情感极性得分
        text_lengths.append(len(text.split()))  # 文本长度
    df['sentiment_score'] = sentiment_scores
    df['text_length'] = text_lengths

add_sentiment_length_features(data)

# 数据增强：对3-4星数据过采样
def augment_3_4_star_reviews(df):
    augmented_texts, augmented_ratings = [], []
    for _, row in df[df['class'] == 1].iterrows():
        augmented_text = augment_text(row['text'])
        augmented_texts.append(augmented_text)
        augmented_ratings.append(row['rating'])
    df_augmented = pd.DataFrame({'text': augmented_texts, 'rating': augmented_ratings})
    df_augmented['class'] = df_augmented['rating'].apply(convert_rating_to_class)
    return pd.concat([df, df_augmented])

data = augment_3_4_star_reviews(data)

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

# 数据集类
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts = df['text'].values
        self.labels = df['class'].values
        self.sentiment_scores = df['sentiment_score'].values
        self.text_lengths = df['text_length'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        sentiment_score = torch.tensor(self.sentiment_scores[idx], dtype=torch.float)
        text_length = torch.tensor(self.text_lengths[idx], dtype=torch.float)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
            'sentiment_score': sentiment_score,
            'text_length': text_length
        }

# 自定义模型
class BERTReviewClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BERTReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 冻结部分BERT层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, input_ids, attention_mask, sentiment_score, text_length):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        extended_features = torch.cat((pooled_output, sentiment_score.unsqueeze(1), text_length.unsqueeze(1)), dim=1)
        return self.classifier(extended_features)

# 过采样权重计算
class_counts = data['class'].value_counts().to_dict()
weights = [1.0 / class_counts[c] for c in data['class']]
sampler = WeightedRandomSampler(weights, len(weights))

# 数据分割
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data, test_size=0.2, stratify=data['class'], random_state=42
)

# 数据加载
train_dataset = ReviewDataset(train_texts, tokenizer, max_len)
val_dataset = ReviewDataset(val_texts, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32)

from tqdm import tqdm  # 导入进度条库

# 训练模型函数
def train_model(model, train_loader, val_loader, epochs=5):  # 减少训练轮次
    # 使用标签平衡的损失函数
    weights = torch.tensor([1.2, 1.0, 0.8]).to(device)  # 根据类别分布调整权重
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 优化器设置
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},  # BERT层使用更小的学习率
        {'params': model.classifier.parameters(), 'lr': 1e-4}  # 分类层使用更大的学习率
    ], weight_decay=0.01)  # 增加权重衰减
    
    # 添加warmup和学习率调度
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        predictions, true_labels = [], []
        
        # 添加进度条
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                
                pbar.update(1)  # 更新进度条
                
        # 计算训练指标
        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(true_labels, predictions)
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_predictions, val_true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                val_predictions.extend(preds.cpu().tolist())
                val_true_labels.extend(labels.cpu().tolist())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_true_labels, val_predictions)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Average train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
        print(f'Average val loss: {avg_val_loss:.4f}, accuracy: {val_acc:.4f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
                
        # 每两个epoch输出一次混淆矩阵
        if epoch % 2 == 0:
            print("\nClassification Report:")
            print(classification_report(val_true_labels, val_predictions,
                                     target_names=['1-2星', '3-4星', '5星']))
            plot_confusion_matrix(val_true_labels, val_predictions, epoch)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练准确率")
    plt.plot(val_accs, label="验证准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.legend()
    
    plt.tight_layout()
    import os
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 保存图像而不是显示
    plt.savefig("figures/loss_curve.png")
    plt.close()
    return train_losses, val_losses, train_accs, val_accs

# 数据准备
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"], data["rating"], 
    test_size=0.2, 
    stratify=data["class"],  # 确保分层采样
    random_state=42
)

# 创建数据集和数据加载器
train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist(), tokenizer, max_len)
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist(), tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 增大batch size
val_loader = DataLoader(val_dataset, batch_size=32)

# 初始化模型并训练
model = BERTReviewClassifier().to(device)
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader)
