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

# 设置Matplotlib的字体参数
plt.rcParams['font.family'] = 'SimHei' # 替换为你选择的字体
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据集，并取前2000条数据
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"][:10000])  # 仅取10000条数据

# 设置BERT的分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

# 定义数据集类
class ReviewDataset(Dataset):
    def __init__(self, texts, ratings):
        self.texts = texts
        self.ratings = [int(rating) - 1 for rating in ratings]  # 将评分转换为类别索引

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
        rating = torch.tensor(self.ratings[idx], dtype=torch.long)  # 使用long类型以适应分类任务
        return input_ids, attention_mask, rating

# 划分训练集和验证集
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"].fillna(""), data["rating"], test_size=0.2, random_state=42
)
train_dataset = ReviewDataset(train_texts.tolist(), train_ratings.tolist())
val_dataset = ReviewDataset(val_texts.tolist(), val_ratings.tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 修改模型结构，将LSTM层数调整为两层
class ReviewLSTM(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256, output_dim=5, dropout_rate=0.3):  # embedding_dim增至128，hidden_dim增至256
        super(ReviewLSTM, self).__init__()
        self.embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        
        # 双层LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=dropout_rate, batch_first=True)
        
        # 增加两层全连接层并加入Dropout
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后一层LSTM的隐藏状态并通过全连接层
        hidden = self.dropout(hidden[-1])
        x = self.fc1(hidden)
        x = self.dropout(nn.ReLU()(x))
        x = self.fc2(x)
        x = self.dropout(nn.ReLU()(x))
        x = self.fc3(x)
        
        return self.softmax(x)


# 初始化模型、损失函数和优化器
model = ReviewLSTM().to(device)  # 将模型移动到 CUDA
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 修改训练函数，增加准确度计算
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
            loss = criterion(outputs, ratings)
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
                loss = criterion(outputs, ratings)
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

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 生成混淆矩阵
def plot_confusion_matrix(model, val_loader):
    model.eval()
    val_predictions, val_labels = [], []

    with torch.no_grad():
        for input_ids, attention_mask, ratings in val_loader:
            input_ids, attention_mask, ratings = input_ids.to(device), attention_mask.to(device), ratings.to(device)
            outputs = model(input_ids, attention_mask)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(ratings.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(val_labels, val_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# 生成并绘制混淆矩阵
plot_confusion_matrix(model, val_loader)
