# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import copy
import pickle  
import time
from sklearn.utils import resample
from collections import Counter
import os
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import threading



# 加载数据集
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
data = pd.DataFrame(dataset["full"])
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
    def __init__(self, texts, ratings, tokenizer, max_len, random_crop_prob=0.3, use_cache=True):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.random_crop_prob = random_crop_prob
        self.use_cache = use_cache
        
        if use_cache:
            # 只缓存原始文本的基础编码
            self.base_encodings = []
            print("Pre-processing base texts...")
            for text in tqdm(texts):
                encoded = self.tokenizer.encode_plus(
                    str(text),
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                self.base_encodings.append({
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze()
                })
            print("Base text pre-processing completed!")

    def random_crop(self, text):
        """随机裁剪文本"""
        words = text.split()
        if len(words) > 50 and random.random() < self.random_crop_prob:
            start_idx = random.randint(0, len(words) - 50)
            end_idx = start_idx + random.randint(30, 50)
            return ' '.join(words[start_idx:end_idx])
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        # 训练模式：应用随机裁剪
        if self.training and random.random() < self.random_crop_prob:
            # 对原始文本进行随机裁剪
            text = self.random_crop(str(self.texts[idx]))
            # 为裁剪后的文本重新编码
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze()
            attention_mask = encoded['attention_mask'].squeeze()
        else:
            # 验证模式或不需要裁剪时：使用缓存的基础编码
            if self.use_cache:
                encoding = self.base_encodings[idx]
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
            else:
                # 不使用缓存时的后备方案
                encoded = self.tokenizer.encode_plus(
                    str(self.texts[idx]),
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].squeeze()
                attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(convert_rating_to_class(float(self.ratings[idx])), dtype=torch.long)
        }
    
    def train(self):
        """设置训练模式"""
        self.training = True
        
    def eval(self):
        """设置评估模式"""
        self.training = False

class EnhancedBERTClassifier(nn.Module):
    def __init__(self, dropout=0.3):  # 降低dropout率
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # 移除特征融合层，简化模型结构
        
        # 保留但简化多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,  # 减少头数
            dropout=0.3
        )
        
        # 简化特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2)
        )
        
        # 分类头使用更深的层次
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        # 解冻BERT参数，但使用差异化学习率
        self._unfreeze_bert_strategically()
        
    def _unfreeze_bert_strategically(self):
        # 后面的层使用更大的学习率
        layers = self.bert.encoder.layer
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = True
    def forward(self, input_ids, attention_mask):
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=True  # 启用注意力输出
        )
    
        hidden_states = outputs.last_hidden_state
        attentions = outputs.attentions  # 获取注意力权重
    
        # 应用多头注意力
        attn_output, _ = self.multihead_attn(
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)
    
        # 特征增强
        enhanced_features = self.feature_enhancement(attn_output)
    
        # 获取[CLS]标记的输出并分类
        cls_output = enhanced_features[:, 0]
        return self.classifier(cls_output)
def plot_final_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    recall = np.diag(cm) / np.sum(cm, axis=1)  # 按行求召回率
    precision = np.diag(cm) / np.sum(cm, axis=0)  # 按列求精准率
    
    # 扩展混淆矩阵以包括召回率和精准率
    cm_with_metrics = np.zeros((cm.shape[0] + 1, cm.shape[1] + 1), dtype=float)
    cm_with_metrics[:cm.shape[0], :cm.shape[1]] = cm
    cm_with_metrics[:-1, -1] = recall  # 添加召回率
    cm_with_metrics[-1, :-1] = precision  # 添加精准率
    
    # 绘制
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_with_metrics, annot=True, fmt=".2f", cmap="Blues")
    plt.title("最终混淆矩阵 (含召回率和精准率)")
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    
    # 添加轴标签
    class_names = [f"类别 {i}" for i in range(cm.shape[0])]
    plt.xticks(ticks=np.arange(len(class_names) + 1) + 0.5, labels=class_names + ["精准率"])
    plt.yticks(ticks=np.arange(len(class_names) + 1) + 0.5, labels=class_names + ["召回率"], rotation=0)
    # 确保保存目录存在
    if not os.path.exists("figures"):
        os.makedirs("figures")
    # 保存图片
    plt.savefig("figures/final_confusion_matrix_with_metrics.png")
    plt.close()



# 首先安装keyboard库：pip install keyboard
import keyboard

# 定义控制器类来控制暂停状态
class TrainingController:
    def __init__(self):
        self.pause_training = False

    def toggle_training(self):
        # 切换暂停状态
        self.pause_training = not self.pause_training
        status = "Paused" if self.pause_training else "Resumed"
        print(f"Training {status}")

# 实例化控制器
controller = TrainingController()

# 注册热键
keyboard.add_hotkey('ctrl+f12', controller.toggle_training)

def train_model(model, train_loader, val_loader, epochs=10, resume=False):
    # 修改损失函数权重，更温和的类别平衡
    weights = torch.tensor([1.05, 1.0, 0.95]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 3e-7},  # 从1e-6降至3e-7
        {'params': model.multihead_attn.parameters(), 'lr': 1.5e-6},  # 从5e-6降至1.5e-6
        {'params': model.feature_enhancement.parameters(), 'lr': 1.5e-6},
        {'params': model.classifier.parameters(), 'lr': 1.5e-6}
    ], weight_decay=0.01)
    
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        eta_min=1e-7  # 设置最小学习率
    )

    start_epoch = 0
    start_batch = 0
    best_model_state = None
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_model_state = None
    best_val_acc = 0.0
   
    if resume:
        checkpoint_data = load_checkpoint(model, optimizer, scheduler)
        if checkpoint_data:
            start_epoch, start_batch, train_losses, val_losses, train_accs, val_accs = checkpoint_data
            print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
            if start_batch >= len(train_loader) - 1:
                start_batch = 0
                start_epoch += 1

    save_interval = 100  # 每100个batch保存和打印一次
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0
            total_samples = 0
            predictions, true_labels = [], []
            
            # 创建带有更多指标的进度条
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    if epoch == start_epoch and batch_idx < start_batch:
                        pbar.update(1)
                        continue

                    while controller.pause_training:
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                     train_losses, val_losses, train_accs, val_accs,
                                     batch_idx, "pause_checkpoint.pth")
                        time.sleep(3)

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    batch_size = input_ids.size(0)
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新loss统计
                    current_loss = loss.item() * batch_size
                    total_loss += current_loss
                    total_samples += batch_size
                    
                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
                    
                    # 计算当前的平均损失和准确率
                    current_avg_loss = total_loss / total_samples
                    current_accuracy = accuracy_score(true_labels, predictions)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # 更新进度条信息
                    pbar.set_postfix({
                        'loss': f'{current_avg_loss:.4f}',
                        'acc': f'{current_accuracy:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                    pbar.update(1)
                    
                    # 每save_interval个batch保存检查点
                    if (batch_idx + 1) % save_interval == 0:
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                     train_losses, val_losses, train_accs, val_accs,
                                     batch_idx)
            
            # 计算整个epoch的训练指标
            avg_train_loss = total_loss / total_samples
            train_acc = accuracy_score(true_labels, predictions)
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0
            total_val_samples = 0
            val_predictions, val_true_labels = [], []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    batch_size = input_ids.size(0)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item() * batch_size
                    total_val_samples += batch_size
                    
                    _, preds = torch.max(outputs, dim=1)
                    val_predictions.extend(preds.cpu().tolist())
                    val_true_labels.extend(labels.cpu().tolist())
            
            avg_val_loss = total_val_loss / total_val_samples
            val_acc = accuracy_score(val_true_labels, val_predictions)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
            print(f'Val loss: {avg_val_loss:.4f}, accuracy: {val_acc:.4f}')
            
        for epoch in range(start_epoch, epochs):
            model.train()
            total_loss = 0
            total_samples = 0  # 添加样本计数
            predictions, true_labels = [], []
            batch_loss = 0  # 用于计算最近100个batch的平均loss
            batch_samples = 0  # 添加batch样本计数
            batch_predictions = []  # 用于计算最近100个batch的准确率
            batch_labels = []
            
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    if epoch == start_epoch and batch_idx < start_batch:
                        pbar.update(1)
                        continue

                    while controller.pause_training:
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                     train_losses, val_losses, train_accs, val_accs,
                                     batch_idx, "pause_checkpoint.pth")
                        time.sleep(3)

                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    batch_size = input_ids.size(0)  # 获取当前batch的实际大小
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # 更新loss统计，考虑实际batch大小
                    current_loss = loss.item() * batch_size
                    batch_loss += current_loss
                    total_loss += current_loss
                    
                    batch_samples += batch_size
                    total_samples += batch_size
                    
                    _, preds = torch.max(outputs, dim=1)
                    batch_predictions.extend(preds.cpu().tolist())
                    batch_labels.extend(labels.cpu().tolist())
                    
                    # 每100个batch打印一次训练指标
                    if (batch_idx + 1) % save_interval == 0:
                        avg_batch_loss = batch_loss / save_interval
                        batch_accuracy = accuracy_score(batch_labels, batch_predictions)
                        
                        print(f"\nEpoch {epoch+1}, Batch {batch_idx+1}")
                        print(f"Train Loss: {avg_batch_loss:.4f}")
                        print(f"Train Accuracy: {batch_accuracy:.4f}")
                        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
                        
                        # 保存检查点
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                     train_losses, val_losses, train_accs, val_accs,
                                     batch_idx)
                        
                        # 重置batch统计
                        batch_loss = 0
                        batch_predictions = []
                        batch_labels = []
                    
                    pbar.update(1)
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({'lr': f'{current_lr:.2e}'})
                    
                    # 累计整体epoch的统计
                    predictions.extend(preds.cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
            
            # 计算整个epoch的训练指标
            avg_train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(true_labels, predictions)
            train_losses.append(avg_train_loss)
            train_accs.append(train_acc)
            
            # 验证阶段
            model.eval()
            total_val_loss = 0
            total_val_samples = 0  # 添加验证样本计数
            val_predictions, val_true_labels = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    total_val_loss += loss.item() * batch_size
                    total_val_samples += batch_size

                    _, preds = torch.max(outputs, dim=1)
                    val_predictions.extend(preds.cpu().tolist())
                    val_true_labels.extend(labels.cpu().tolist())
            
            avg_val_loss = total_val_loss / total_val_samples  # 使用验证集总样本数
            val_acc = accuracy_score(val_true_labels, val_predictions)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                if not os.path.exists("savepoint"):
                    os.makedirs("savepoint")
                torch.save(best_model_state, 'savepoint/best_model.pth')
            
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Average train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
            print(f'Average val loss: {avg_val_loss:.4f}, accuracy: {val_acc:.4f}')
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
                if epoch == epochs - 1:
                    plot_final_confusion_matrix(val_true_labels, val_predictions)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    plot_final_confusion_matrix(val_true_labels, val_predictions)
                    break

    except Exception as e:
        print(f"Training interrupted: {str(e)}")
        save_checkpoint(epoch, model, optimizer, scheduler,
                       train_losses, val_losses, train_accs, val_accs,
                       batch_idx, "error_checkpoint.pth")
        raise e

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

    model.load_state_dict(best_model_state)
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

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses, train_accs, val_accs, batch_idx, filename="checkpoint.pth"):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    if not os.path.exists("savepoint"):
        os.makedirs("savepoint")
    torch.save(checkpoint, "savepoint/%s"%filename)
    
    print(f"\nCheckpoint saved: Epoch {epoch}, Batch {batch_idx}")

def load_checkpoint(model, optimizer, scheduler, filename="checkpoint.pth"):
    """加载检查点"""
    # 构建完整的检查点路径
    checkpoint_path = os.path.join("savepoint", filename)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state loaded")
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
        
        # 加载调度器状态
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
        
        # 返回训练状态
        return (checkpoint['epoch'], 
                checkpoint['batch_idx'], 
                checkpoint['train_losses'], 
                checkpoint['val_losses'],
                checkpoint['train_accs'], 
                checkpoint['val_accs'])
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return None
  
def add_text_augmentation(text):
    """
    对文本进行轻微增强，避免过采样导致的过拟合
    """
    if np.random.random() < 0.25:  # 25%的概率进行增强
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
            weights.append(1.8)  # 给予更高的权重
        elif label == 0:
            weights.append(1.2)
        elif label == 2:
            weights.append(1.1)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

def prepare_data_loaders(data, tokenizer, max_len, batch_size=32, cache_dir='cached_data'):
    """准备数据加载器，支持缓存和动态增强"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'processed_data.pkl')
    
    if os.path.exists(cache_file):
        print("Loading cached processed data...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            train_dataset = cached_data['train_dataset']
            val_dataset = cached_data['val_dataset']
        print("Cache loaded successfully!")
    else:
        print("Processing data and creating new cache...")
        balanced_data = balanced_sampling_strategy(data)
        
        train_texts, val_texts, train_ratings, val_ratings = train_test_split(
            balanced_data["text"], 
            balanced_data["rating"],
            test_size=0.2,
            stratify=balanced_data["class"],
            random_state=42
        )
        
        train_dataset = ReviewDataset(
            train_texts.tolist(), 
            train_ratings.tolist(), 
            tokenizer, 
            max_len,
            use_cache=True
        )
        train_dataset.train()  # 设置为训练模式
        
        val_dataset = ReviewDataset(
            val_texts.tolist(), 
            val_ratings.tolist(), 
            tokenizer, 
            max_len,
            use_cache=True
        )
        val_dataset.eval()  # 设置为评估模式
        
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'train_dataset': train_dataset,
                'val_dataset': val_dataset
            }, f)
        print("Data processed and cached successfully!")
    
    # 确保设置正确的模式
    train_dataset.train()
    val_dataset.eval()
    
    # 创建采样器和数据加载器
    train_sampler = create_weighted_sampler(train_dataset)
    
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
    

def visualize_attention_weights(model, texts, tokenizer):
    i=0
    for text in texts:
        i+=1
        model.eval()
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            # 从 BERT 模型获取注意力权重
            outputs = model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # 获取最后一层的注意力权重
            last_layer_attn = outputs.attentions[-1]  # 取最后一层的注意力权重
            attention = last_layer_attn[:, 0, :].mean(dim=1).squeeze()  # 平均所有头并移除多余维度
            
        # 将 attention 转换为一维，并确认其长度是否与 tokens 一致
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        attention = attention[:len(tokens)]  # 截断或对齐长度
        print("Attention shape:", attention.shape)
        print("Tokens length:", len(tokens))
        
        plt.figure(figsize=(15, 5))
        sns.barplot(x=tokens, y=attention.cpu().numpy())
        plt.xticks(rotation=45, ha='right')
        plt.title('Attention Weights Visualization')
        plt.tight_layout()
        plt.savefig('figures/attention_visualization%s.png'%i)
        plt.close()


def test_model_predictions(model, texts, tokenizer):
    model.eval()
    results = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
            
        sentiment = ['Negative', 'Neutral', 'Positive'][prediction]
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': f"{confidence:.2%}"
        })
    
    return results

# 使用改进后的数据加载器
train_loader, val_loader = prepare_data_loaders(data, tokenizer, max_len)
model = EnhancedBERTClassifier().to(device)
train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader,resume=True, batch_size=48)



# 示例使用
test_texts = [
    "This product is absolutely amazing! I love everything about it.",
    "It's okay, but could be better. Some pros and cons.",
    "Terrible product, waste of money. Would not recommend."
]

# 加载训练好的模型
model = EnhancedBERTClassifier().to(device)
model.load_state_dict(torch.load('best_model.pth'))
# 测试预测
results = test_model_predictions(model, test_texts, tokenizer)
for result in results:
    print(f"\nText: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
# 可视化测试文本的注意力权重
visualize_attention_weights(model, test_texts, tokenizer)


