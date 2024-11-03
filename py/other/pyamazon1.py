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
# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei' # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
# 加载数据集，并取前2000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:2000])  # 仅取2000条数据
print(data.head())

# 初始化BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

# 定义二分类数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings):
        self.texts = texts
        # 评分 4 和 5 作为好评 (1)，4 分以下作为差评 (0)
        self.ratings = [1 if rating >= 4 else 0 for rating in ratings]

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
        rating = torch.tensor(self.ratings[idx], dtype=torch.float)
        return input_ids, attention_mask, rating

# 划分训练集和验证集
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"].fillna(""), data["rating"], test_size=0.2, random_state=42
)
train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist())
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 定义二分类LSTM模型
class ReviewLSTM(nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, output_dim=1):
        super(ReviewLSTM, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        output = self.sigmoid(self.fc(hidden[-1]))
        return output

# 初始化模型、损失函数和优化器
model = ReviewLSTM()
criterion = nn.BCELoss()  # 使用二分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 定义训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, patience=3):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for input_ids, attention_mask, ratings in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask).squeeze()
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证模型
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, ratings in val_loader:
                outputs = model(input_ids, attention_mask).squeeze()
                loss = criterion(outputs, ratings)
                val_loss += loss.item()
                
                # 计算准确率
                predictions = (outputs >= 0.5).float()
                correct += (predictions == ratings).sum().item()
                total += ratings.size(0)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

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
    plt.plot(val_accuracies, label="验证准确度")
    plt.xlabel("轮次")
    plt.ylabel("准确度")
    plt.legend()
    plt.show()

# 开始训练
train_model(model, train_loader, val_loader, criterion, optimizer)
