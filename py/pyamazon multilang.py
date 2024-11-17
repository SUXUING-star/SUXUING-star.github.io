# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
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
import keyboard
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
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
  
# 加载多语言数据集
def load_multilingual_dataset():
    # 加载原有的英文数据集
    en_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    en_data = pd.DataFrame(en_dataset["full"][:40])  # 原有英文数据

    # 额外加载中文数据集
    zh_dataset = load_dataset("SetFit/amazon_reviews_multi_zh")
    zh_data = pd.DataFrame({
        'text': zh_dataset['train']['review_body'],
        'label': zh_dataset['train']['stars']
    }[:40])

    # 合并中英文数据
    data = pd.concat([en_data, zh_data], ignore_index=True)
    return data
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

class MultilingualReviewDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_len, random_crop_prob=0.3, use_cache=True):
        self.texts = texts
        self.ratings = ratings
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.random_crop_prob = random_crop_prob
        self.use_cache = use_cache
        self.training = True
        
        if use_cache:
            self.base_encodings = []
            print("预处理基础文本...")
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
            print("基础文本预处理完成！")

    def random_crop(self, text):
        """随机裁剪文本，支持中英文"""
        if isinstance(text, str):
            if len(text) > 50 and random.random() < self.random_crop_prob:
                start_idx = random.randint(0, len(text) - 50)
                end_idx = start_idx + random.randint(30, 50)
                return text[start_idx:end_idx]
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.training and random.random() < self.random_crop_prob:
            text = self.random_crop(str(self.texts[idx]))
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
            if self.use_cache:
                encoding = self.base_encodings[idx]
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']
            else:
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
        self.training = True
        
    def eval(self):
        self.training = False

class MultilingualBERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(MultilingualBERTClassifier, self).__init__()
        # 使用多语言BERT模型
        self.bert = AutoModel.from_pretrained('bert-base-multilingual-cased')
        hidden_size = self.bert.config.hidden_size
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.3
        )
        
        self.feature_enhancement = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2)
        )
        
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
        
        self._unfreeze_bert_strategically()
        
    def _unfreeze_bert_strategically(self):
        layers = self.bert.encoder.layer
        for i, layer in enumerate(layers):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=True
        )
    
        hidden_states = outputs.last_hidden_state
        attentions = outputs.attentions
    
        attn_output, _ = self.multihead_attn(
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            hidden_states.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)
    
        enhanced_features = self.feature_enhancement(attn_output)
        cls_output = enhanced_features[:, 0]
        return self.classifier(cls_output)
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
            weights.append(1.8)  # 给予更高的权重
        elif label == 0:
            weights.append(1.2)
        elif label == 2:
            weights.append(1.0)
    
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
   
def prepare_multilingual_data():
    # 加载数据
    data = load_multilingual_dataset()
    
    # 初始化多语言分词器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # 准备数据加载器
    train_loader, val_loader = prepare_data_loaders(data, tokenizer, max_len=128)
    
    return train_loader, val_loader, tokenizer

# 保持原有的其他功能函数不变
def convert_rating_to_class(rating):
    if rating <= 2:
        return 0
    elif rating <= 4:
        return 1
    return 2
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
    weights = torch.tensor([1.1, 1.0, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 使用分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},  # 降低BERT的学习率
        {'params': model.multihead_attn.parameters(), 'lr': 5e-5},
        {'params': model.feature_enhancement.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-5}
    ], weight_decay=0.02)  # 增加权重衰减
    
    # 计算总训练步数
    num_training_steps = len(train_loader) * epochs
    # 设置热身步数为总步数的5%
    num_warmup_steps = num_training_steps // 10
    
    # 使用get_cosine_schedule_with_warmup替换原来的线性调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # 初始化训练状态变量
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
            # 从上次保存的轮次开始训练
            start_epoch = start_epoch
            # 如果是最后一个batch,从下一个epoch开始
            if start_batch >= len(train_loader) - 1:
                start_batch = 0
                start_epoch += 1
    # 添加自动保存间隔
    save_interval = 100  # 每100个batch保存一次
    try:
        for epoch in range(start_epoch, epochs):
            # 训练阶段
            model.train()
            total_loss = 0
            predictions, true_labels = [], []
            
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    # 如果是恢复训练且在第一个epoch，跳过已经训练过的batch
                    if epoch == start_epoch and batch_idx < start_batch:
                        pbar.update(1)
                        continue
                    # 检查是否需要暂停
                    while controller.pause_training:
                        # 暂停时保存检查点
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                        train_losses, val_losses, train_accs, val_accs,
                                        batch_idx, "pause_checkpoint.pth")
                        time.sleep(3)
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
                    # 定期保存检查点
                    if batch_idx % save_interval == 0:
                        save_checkpoint(epoch, model, optimizer, scheduler,
                                     train_losses, val_losses, train_accs, val_accs,
                                     batch_idx)
                    
                    pbar.update(1)
                    
                    # 更新进度条显示当前学习率
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({'lr': f'{current_lr:.2e}'})
            
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
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                # 确保保存目录存在
                if not os.path.exists("savepoint"):
                    os.makedirs("savepoint")
                torch.save(best_model_state, 'savepoint/best_model.pth')
            
            print(f'\nEpoch {epoch+1}/{epochs}')
            print(f'Average train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
            print(f'Average val loss: {avg_val_loss:.4f}, accuracy: {val_acc:.4f}')
            print(f'Current learning rate: {scheduler.get_last_lr()[0]:.2e}')
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pth')
                # 保存最后一次的混淆矩阵
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
        # 发生异常时保存检查点
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

    # 训练完成后加载最佳模型
    model.load_state_dict(best_model_state)
    return train_losses, val_losses, train_accs, val_accs


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
# 使用示例
def main():
    print(f"正在准备中英文的模型和数据...")
    
    # 准备数据
    train_loader, val_loader, tokenizer = prepare_multilingual_data()
    
    # 初始化模型
    model = MultilingualBERTClassifier().to(device)
    
    # 训练模型
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, epochs=10, resume=False
    )
    
    # 测试示例
    test_texts = [
        "这个产品太棒了！我非常喜欢。",
        "产品质量一般，有优点也有缺点。",
        "很差劲的产品，完全浪费钱。",
        "This product is amazing! I love it.",
        "It's okay, has both pros and cons.",
        "Terrible product, waste of money."
    ]
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 测试预测
    results = test_model_predictions(model, test_texts, tokenizer)
    for result in results:
        print(f"\n文本: {result['text']}")
        print(f"情感: {result['sentiment']}")
        print(f"置信度: {result['confidence']}")
    
    # 可视化注意力权重
    visualize_attention_weights(model, test_texts, tokenizer)

if __name__ == "__main__":
    main()  # 或 'en' 用于英文