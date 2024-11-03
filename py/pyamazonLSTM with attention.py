# 导入所需库
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据集，并取前10000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:20000])
print(data.head())

# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

class ReviewDataset(Dataset):
    def __init__(self, texts, ratings):
        self.texts = texts
        self.ratings = ratings

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        # 将评分转换为类别索引（1-5 → 0-4）
        rating = torch.tensor(int(self.ratings[idx]) - 1, dtype=torch.long)
        return input_ids, attention_mask, rating

# 增强版LSTM分类模型
class EnhancedReviewClassifier(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=256, num_layers=2, dropout=0.3, bidirectional=True, num_classes=5):
        super(EnhancedReviewClassifier, self).__init__()
        
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 全连接层
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)  # 输出改为5个类别
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 添加softmax层

    def forward(self, input_ids, attention_mask):
        # 词嵌入
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM层
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        attention_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 全连接层
        x = self.dropout(attention_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # 不在这里应用softmax，因为CrossEntropyLoss已包含
        
        return x

# 数据预处理和加载
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"].fillna(""), data["rating"], test_size=0.2, random_state=42
)

train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist())
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 初始化模型、损失函数和优化器
model = EnhancedReviewClassifier().to(device)
criterion = nn.CrossEntropyLoss()  # 改用CrossEntropyLoss
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.show()

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=5):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0
        train_preds = []
        train_true = []
        
        for input_ids, attention_mask, ratings in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, ratings)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
            
            # 收集预测结果
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(ratings.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for input_ids, attention_mask, ratings in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                ratings = ratings.to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(ratings.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # 每个epoch结束时打印详细的分类报告
        if epoch % 5 == 0:  # 每5个epoch打印一次详细报告
            print("\nClassification Report:")
            print(classification_report(val_true, val_preds, 
                                     target_names=['1 星', '2 星', '3 星', '4 星', '5 星']))
            plot_confusion_matrix(val_true, val_preds)
            
        print("-" * 50)

        # 学习率调整
        scheduler.step(val_loss)

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

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
    plt.show()

    return train_losses, val_losses, train_accs, val_accs

# 开始训练
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)