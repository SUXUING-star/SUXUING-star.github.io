# 导入所需库
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据集
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:200000])
# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 数据预处理：添加文本清理和长度限制
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # 限制文本长度，避免特别长的评论
    return str(text)[:512]

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

class ReviewDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_len):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 添加动态裁剪
        if len(text.split()) > 100:  # 如果文本太长
            text = ' '.join(text.split()[:100])  # 只取前100个词
            
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(convert_rating_to_class(float(self.ratings[idx])), dtype=torch.long)
        }

class BERTReviewClassifier(nn.Module):
    def __init__(self, dropout=0.5):  # 增加dropout率
        super(BERTReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 冻结BERT底层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(8):  # 冻结前8层
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
        
        hidden_size = self.bert.config.hidden_size
        
        # 改进分类头：添加BatchNorm和更多的Dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # 添加权重初始化
        self._init_weights()
        
    def _init_weights(self):
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
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
