from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei' # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 加载数据集，并取前2000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"])  # 仅取1000条数据

# 设置BERT的分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

# 定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings):
        self.texts = texts
        # 将评分转换为类别索引（1-5对应0-4）
        self.ratings = [int(rating) - 1 for rating in ratings]

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
        rating = torch.tensor(self.ratings[idx], dtype=torch.long)
        return input_ids, attention_mask, rating

# 划分训练集和验证集
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"].fillna(""), data["rating"], test_size=0.2, random_state=42
)
train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist())
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class ReviewLSTM(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=256, output_dim=5, dropout=0.3):
        super(ReviewLSTM, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        # 设置双向 LSTM，单层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 注意：双向LSTM的输出是2倍隐藏维度
        self.dropout = nn.Dropout(dropout)  # 增加Dropout层
        self.softmax = nn.Softmax(dim=1)

        # 初始化权重
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        hidden = lstm_out[:, -1, :]  # 取最后时刻的输出
        output = self.fc(hidden)
        return self.softmax(output)

# 初始化模型、损失函数和优化器
model = ReviewLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型加载到GPU
model = ReviewLSTM().to(device)

# 修改训练函数，支持GPU训练和准确度计算
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=3):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        train_predictions, train_labels = [], []
        
        for input_ids, attention_mask, ratings in train_loader:
            input_ids, attention_mask, ratings = input_ids.to(device), attention_mask.to(device), ratings.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, ratings.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            train_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(ratings.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证模型
        model.eval()
        val_loss = 0
        val_predictions, val_labels = [], []
        
        with torch.no_grad():
            for input_ids, attention_mask, ratings in val_loader:
                input_ids, attention_mask, ratings = input_ids.to(device), attention_mask.to(device), ratings.to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, ratings.long())
                val_loss += loss.item()

                val_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(ratings.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, "
              f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("早停法：验证损失未改善，提前停止训练。")
                break

    # 绘制损失和准确度曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="训练准确度")
    plt.plot(val_accuracies, label="验证准确度")
    plt.xlabel("轮次")
    plt.ylabel("准确度")
    plt.legend()
    plt.show()

# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer)