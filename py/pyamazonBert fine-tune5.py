# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import precision_recall_fscore_support,accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup,BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from sklearn.utils import resample
from collections import Counter
import os

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
class DynamicWeightAdjuster:
    def __init__(self, initial_weights, beta=0.9):
        # 确保初始权重是 Float32 类型
        self.current_weights = initial_weights.float()  # 添加 .float()
        self.beta = beta
        self.historical_f1 = {i: [] for i in range(len(initial_weights))}
        
    def update_weights(self, y_true, y_pred):
        from sklearn.metrics import f1_score
        
        f1_scores = f1_score(y_true, y_pred, average=None)
        
        for i, f1 in enumerate(f1_scores):
            self.historical_f1[i].append(f1)
            
        avg_f1_scores = []
        for i in range(len(self.current_weights)):
            if len(self.historical_f1[i]) > 0:
                avg_f1 = sum(self.historical_f1[i][-5:]) / len(self.historical_f1[i][-5:])
                avg_f1_scores.append(avg_f1)
            else:
                avg_f1_scores.append(1.0)
        
        avg_f1_scores = np.array(avg_f1_scores)
        new_weights = 1 / (avg_f1_scores + 1e-5)
        new_weights = new_weights / new_weights.sum() * len(new_weights)
        
        # 确保新权重是 Float32 类型
        self.current_weights = (self.beta * self.current_weights + 
                              (1 - self.beta) * torch.tensor(new_weights, dtype=torch.float32))
        
        return self.current_weights
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
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(15, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.title("训练和验证损失曲线")
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练准确率")
    plt.plot(val_accs, label="验证准确率")
    plt.xlabel("轮次")
    plt.ylabel("准确率")
    plt.title("训练和验证准确率曲线")
    plt.legend()
    
    plt.tight_layout()
    import os
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")
    plt.savefig("figures/training_curves.png")
    plt.close()
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
def train_model(model, train_loader, val_loader, epochs=5):
    # 初始化时确保权重是 Float32 类型
    initial_weights = torch.tensor([1.1, 1.1, 0.8], dtype=torch.float32)
    weight_adjuster = DynamicWeightAdjuster(initial_weights)
    
    # 确保损失函数的权重是 Float32 类型
    criterion = nn.CrossEntropyLoss(weight=weight_adjuster.current_weights.float().to(device))
    
    # 优化器设置
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    # 设置余弦退火学习率调度器
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5  # 半个余弦周期
    )
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                criterion.weight = weight_adjuster.current_weights.float().to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                
                # 使用当前权重计算损失
                criterion.weight = weight_adjuster.current_weights.to(device)
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                
                # 更新进度条
                pbar.update(1)
                
            # 在每个epoch结束时更新类别权重
            new_weights = weight_adjuster.update_weights(true_labels, predictions)
            print(f"\nUpdated class weights: {new_weights}")
        
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
        
        # 打印详细的评估指标
        print("\nValidation Performance by Class:")
        for i, (precision, recall, f1, support) in enumerate(
            zip(*precision_recall_fscore_support(val_true_labels, val_predictions))):
            print(f"Class {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'class_weights': weight_adjuster.current_weights,
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
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
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return train_losses, val_losses, train_accs, val_accs
# 数据准备
train_texts, val_texts, train_ratings, val_ratings = train_test_split(
    data["text"], data["rating"], 
    test_size=0.2, 
    stratify=data["class"],  # 确保分层采样
    random_state=42
)

def balanced_sampling_strategy(data):
    """修改后的平衡采样策略"""
    data = data.copy()
    data['class'] = data['rating'].apply(convert_rating_to_class)
    class_counts = Counter(data['class'])
    
    ## 使用最大类别数量作为基准，而不是平均值
    #max_samples = max(class_counts.values())
    ## 设置一个合理的上限，避免过度采样
    #target_samples = min(max_samples, 50000)  
    # 计算目标采样数量：将中间类别(class=1)的样本数提升到与最多的类别相近
    target_samples = max(class_counts.values())
    
    balanced_dfs = []
    for class_label in class_counts.keys():
        class_data = data[data['class'] == class_label]
        
        if class_label == 1:  # 3-4星评分
            # 采样数量不低于原始数量
            n_samples = max(len(class_data), target_samples)
            resampled = resample(class_data,
                               n_samples=n_samples,
                               random_state=42)
            resampled['text'] = resampled['text'].apply(add_text_augmentation)
            
        elif class_label == 0:  # 1-2星评分
            # 对少数类进行更多的采样
            n_samples = max(len(class_data), int(target_samples * 1.1))
            resampled = resample(class_data,
                               n_samples=n_samples,
                               random_state=42)
            resampled['text'] = resampled['text'].apply(
                lambda x: add_text_augmentation(x, is_negative=True))
            
        else:  # 5星评分
            # 保持原有数量或轻微下采样
            n_samples = min(len(class_data), target_samples)
            resampled = class_data.sample(n=n_samples, random_state=42)
        
        balanced_dfs.append(resampled)
    
    balanced_data = pd.concat(balanced_dfs, ignore_index=True)
    
    # 打印数据分布信息
    print("\nClass distribution after balancing:")
    print(balanced_data['class'].value_counts())
    print(f"Total samples: {len(balanced_data)}")
    
    return balanced_data

def add_text_augmentation(text, is_negative=False):
    if np.random.random() < 0.3:
        if is_negative:
            # 针对1-2星评分的增强方法
            augmentation_methods = [
                lambda x: x.replace('!', '.'),  # 平和的语气
                lambda x: 'Unfortunately, ' + x,  # 添加负面前缀
                lambda x: x + ' However, there might be room for improvement.',
                lambda x: x.replace('terrible', 'not satisfactory').replace('horrible', 'disappointing'),
                lambda x: 'I had issues with this product. ' + x
            ]
        else:
            # 原有的增强方法
            augmentation_methods = [
                lambda x: x.replace('.', '!'),
                lambda x: x + ' Overall, this product is okay.',
                lambda x: x + ' It has both pros and cons.',
                lambda x: 'In my experience, ' + x,
                lambda x: x.replace('good', 'decent').replace('bad', 'not ideal')
            ]
        return np.random.choice(augmentation_methods)(text)
    return text
def create_weighted_sampler(dataset):
    labels = [item['labels'].item() for item in dataset]
    class_counts = Counter(labels)
    
    # 更平衡的权重分配
    weights = []
    for item in dataset:
        label = item['labels'].item()
        if label == 1:  # 3-4星
            weights.append(1.5)  # 降低中间类别的权重
        elif label == 0:  # 1-2星
            weights.append(1.5)  # 提高低星级的权重
        else:  # 5星
            weights.append(1.0)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
# 修改数据加载部分
def prepare_data_loaders(data, tokenizer, max_len, batch_size=32):
    """改进的数据加载器准备函数"""
    # 首先进行训练集验证集划分
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        stratify=data["class"],
        random_state=42
    )
    
    # 只对训练集进行平衡采样和增强
    balanced_train_data = balanced_sampling_strategy(train_data)
    
    # 验证集保持原样
    train_dataset = ReviewDataset(
        balanced_train_data["text"].tolist(), 
        balanced_train_data["rating"].tolist(), 
        tokenizer, 
        max_len
    )
    
    val_dataset = ReviewDataset(
        val_data["text"].tolist(), 
        val_data["rating"].tolist(), 
        tokenizer, 
        max_len
    )
    
    # 创建加权采样器
    train_sampler = create_weighted_sampler(train_dataset)
    
    # 打印数据集大小信息
    print(f"\nTraining set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
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
model = BERTReviewClassifier().to(device)
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader)