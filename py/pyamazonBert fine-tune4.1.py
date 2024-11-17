# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from collections import Counter
import os
import re
from nltk.tokenize import word_tokenize
import nltk
import random
nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:200000])
# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

def preprocess_text(text):
    if pd.isna(text):
        return ""
    # 1. 移除HTML标签
    text = re.sub('<[^<]+?>', '', text)
    # 2. 统一转换为小写
    text = text.lower()
    # 3. 移除特殊字符,但保留一些有意义的标点
    text = re.sub(r'[^\w\s!?,.]', ' ', text)
    # 4. 分词
    words = word_tokenize(text)
    # 5. 移除超短词
    words = [w for w in words if len(w) > 1]
    # 6. 重组文本
    text = ' '.join(words)
    return text[:512]

# 在数据预处理时使用
data['text'] = data['text'].apply(preprocess_text)

# 使用stratified split确保标签分布一致
def convert_rating_to_class(rating):
    if rating <= 2:
        return 0
    elif rating <= 4:
        return 1
    return 2

data['class'] = data['rating'].apply(convert_rating_to_class)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128  # 减小序列长度以降低复杂度

def focal_loss(model, input_ids, attention_mask, labels, alpha=0.25, gamma=2):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 计算 focal loss
    pt = F.softmax(outputs, dim=1)[range(len(labels)), labels]
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    
    return loss.mean()

class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_len):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)  # 返回数据集的长度
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 动态采样策略：较长文本随机截取一个片段
        words = text.split()
        if len(words) > 100:
            start_idx = random.randint(0, len(words) - 100)
            text = ' '.join(words[start_idx:start_idx + 100])
        
        # 添加特殊标记以增强特征提取
        text = f"[CLS] 评论: {text} [SEP]"
            
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True  # 添加token类型信息
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'token_type_ids': encoded['token_type_ids'].squeeze(),
            'labels': torch.tensor(convert_rating_to_class(float(self.ratings[idx])), dtype=torch.long)
        }
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # Feature fusion layer
        self.feature_fusion = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(4)
        ])
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # Feature enhancement layer
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.
                weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        # 1. Get BERT outputs from all layers
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 2. Multi-layer feature fusion
        last_hidden_states = outputs.hidden_states[-4:]
        fused_features = []
        for i, hidden_state in enumerate(last_hidden_states):
            fused_features.append(self.feature_fusion[i](hidden_state))
        
        # 3. Weighted feature fusion
        fused_output = torch.stack(fused_features).mean(dim=0)
        
        # 4. Apply multi-head attention
        attn_output, _ = self.multihead_attn(
            fused_output.transpose(0, 1),
            fused_output.transpose(0, 1),
            fused_output.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)
        
        # 5. Feature enhancement
        enhanced_features = self.feature_enhancement(attn_output)
        
        # 6. Get [CLS] token output and classify
        cls_output = enhanced_features[:, 0]
        return self.classifier(cls_output)
def plot_confusion_matrix(y_true, y_pred,epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 保存图像而不是显示
    plt.savefig("figures/epoch%s-matrix.png"%epoch)
    plt.close()

from tqdm import tqdm  # 导入进度条库

# 训练模型函数
def train_model(model, train_loader, val_loader, epochs=10):
    # 使用标签平衡的损失函数
    weights = torch.tensor([1.2, 0.9, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 优化器设置
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.feature_fusion.parameters(), 'lr': 2e-4},
        {'params': model.multihead_attn.parameters(), 'lr': 2e-4},
        {'params': model.feature_enhancement.parameters(), 'lr': 2e-4},
        {'params': model.classifier.parameters(), 'lr': 2e-4}
    ], weight_decay=0.01)
    
    # 学习率调度
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
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                outputs = model(input_ids, attention_mask)
                #动态平衡权重
                loss = focal_loss(model, input_ids, attention_mask, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                
                pbar.update(1)
                
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
    if not os.path.exists("figures"):
        os.makedirs("figures")
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


def balanced_sampling_strategy(data):
    """
    对数据集进行平衡采样，特别关注中间评分(3-4星)的样本
    """
    # 转换评分为类别
    data = data.copy()
    data['class'] = data['rating'].apply(convert_rating_to_class)
    
    # 统计各类别数量
    class_counts = Counter(data['class'])
    
    # 计算目标采样数量：将中间类别(class=1)的样本数提升到与最多的类别相近
    target_samples = max(class_counts.values())
    
    # 对每个类别进行重采样
    balanced_dfs = []
    for class_label in class_counts.keys():
        class_data = data[data['class'] == class_label]
        
        if class_label == 1:  # 对于3-4星评分
            # 过采样至目标数量，并添加少量噪声避免过拟合
            resampled = resample(class_data,
                               n_samples=int(target_samples * 1.1),  # 稍微过采样
                               random_state=42)
            # 对文本添加轻微变化
            resampled['text'] = resampled['text'].apply(add_text_augmentation)
        else:
            # 其他类别保持原样或轻微下采样
            n_samples = min(len(class_data), target_samples)
            resampled = resample(class_data,
                               n_samples=n_samples,
                               random_state=42)
        
        balanced_dfs.append(resampled)
    
    # 合并平衡采样后的数据
    balanced_data = pd.concat(balanced_dfs, ignore_index=True)
    
    # 保存增强后的数据
    os.makedirs('backup', exist_ok=True)
    balanced_data.to_csv('backup/augmented_balanced_data.csv', index=False)
    
    return balanced_data

def add_text_augmentation(text):
    """
    对文本进行轻微增强，避免过采样导致的过拟合
    """
    if np.random.random() < 0.3:  # 30%的概率进行增强
        augmentation_methods = [
            lambda x: x.replace('.', '!'),  # 改变标点
            lambda x: x + ' Overall, this product is okay.',  # 添加中性评价
            lambda x: x + ' It has both pros and cons.',  # 添加平衡评价
            lambda x: 'In my experience, ' + x,  # 添加前缀
            lambda x: x.replace('good', 'decent').replace('bad', 'not ideal')  # 替换极性词
        ]
        return np.random.choice(augmentation_methods)(text)
    return text

def create_weighted_sampler(dataset):
    """
    创建加权采样器，使模型在训练时更关注中间评分样本
    """
    labels = [item['labels'].item() for item in dataset]
    class_counts = Counter(labels)
    
    # 计算类别权重，中间类别给予更高权重
    weights = []
    for item in dataset:
        label = item['labels'].item()
        if label == 1:  # 3-4星评分类别
            weights.append(1.6)  # 给予更高的权重
        elif label == 0:
            weights.append(1.2)
        elif label == 2:
            weights.append(1.0)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

# 修改数据加载部分
def prepare_data_loaders(data, tokenizer, max_len, batch_size=32):
    """
    准备经过改进的数据加载器
    """
    # 应用平衡采样策略
    balanced_data = balanced_sampling_strategy(data)
    
    # 分割数据集
    train_texts, val_texts, train_ratings, val_ratings = train_test_split(
        balanced_data["text"], 
        balanced_data["rating"],
        test_size=0.2,
        stratify=balanced_data["class"],
        random_state=42
    )
    
    # 创建数据集
    train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist(), tokenizer, max_len)
    val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist(), tokenizer, max_len)
    
    # 创建加权采样器
    train_sampler = create_weighted_sampler(train_dataset)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    
    return train_loader, val_loader

# 使用改进后的数据加载器
train_loader, val_loader = prepare_data_loaders(data, tokenizer, max_len)

# 继续使用原有的模型训练代码
model = EnhancedBERTClassifier().to(device)
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader)