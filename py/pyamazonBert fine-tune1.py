# 导入所需库
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据集，并取前20000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:200000])
print(data.head())

# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 使用BERT tokenizer
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
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 合并标签
        original_rating = int(self.ratings[idx])
        if original_rating <= 2:
            rating = 0  # 1-2 星合并为 0 类
        elif original_rating <= 4:
            rating = 1  # 3-4 星合并为 1 类
        else:
            rating = 2  # 5 星为 2 类
        return input_ids, attention_mask, torch.tensor(rating, dtype=torch.long)

# BERT分类器模型
class BERTReviewClassifier(nn.Module):
    def __init__(self, dropout=0.3, num_classes=3):
        super(BERTReviewClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # BERT的隐藏层大小
        hidden_size = self.bert.config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用[CLS]标记的输出进行分类
        pooled_output = outputs.pooler_output
        
        # 通过分类头
        logits = self.classifier(pooled_output)
        
        return logits

# 数据预处理和加载
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"].fillna(""), data["rating"], test_size=0.2, random_state=42
)

train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist())
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小batch size
val_loader = DataLoader(val_dataset, batch_size=16)

# 初始化模型、损失函数和优化器
model = BERTReviewClassifier().to(device)
criterion = nn.CrossEntropyLoss()

# 设置不同的学习率
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=3):
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
        if epoch % 2 == 0:  # 每2个epoch打印一次详细报告
            print("\nClassification Report:")
            print(classification_report(val_true, val_preds, 
                                     target_names=['1-2星', '3-4星', '5星']))
            plot_confusion_matrix(val_true, val_preds,epoch)
            
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
    import os
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # 保存图像而不是显示
    plt.savefig("figures/loss_curve.png")
    plt.close()

    return train_losses, val_losses, train_accs, val_accs

# 开始训练
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)