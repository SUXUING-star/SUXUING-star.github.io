# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import gc
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import copy
import os
import random
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import re
# 在文件开头添加
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# NLTK下载只执行一次
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('wordnet')

download_nltk_data()
from nltk.corpus import wordnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

class TextAugmenter:
    @staticmethod
    def synonym_replacement(text, n=2):  # 增加替换数量
        words = word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum() and len(word) > 3]))  # 只替换较长的词
        n = min(n, len(random_word_list))
        
        for _ in range(n):
            random_word = random.choice(random_word_list)
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    if lemma.name() != random_word:  # 避免选择相同的词
                        synonyms.append(lemma.name())
            if len(synonyms) > 0:
                synonym = random.choice(list(set(synonyms)))
                try:
                    idx = new_words.index(random_word)
                    new_words[idx] = synonym
                except ValueError:
                    continue
        
        return ' '.join(new_words)

    @staticmethod
    def random_deletion(text, p=0.15):  # 略微增加删除概率
        words = word_tokenize(text)
        if len(words) <= 3:  # 保护短文本
            return text
        
        new_words = []
        for word in words:
            if random.random() > p or len(new_words) < 3:  # 确保至少保留3个词
                new_words.append(word)
        
        if len(new_words) == 0:
            return text
            
        return ' '.join(new_words)

    @staticmethod
    def random_swap(text, n=2):  # 增加交换次数
        words = word_tokenize(text)
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return ' '.join(new_words)

    @staticmethod
    def back_translation(text):
        # 模拟回译效果，通过同义词替换和词序调整
        words = word_tokenize(text)
        # 随机调整词序
        if len(words) > 3:
            split_point = random.randint(1, len(words)-2)
            words = words[split_point:] + words[:split_point]
        return ' '.join(words)

class ReviewGenerationDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_len=256, augment=False):  # 增加最大长度
        self.texts = texts.tolist()
        self.ratings = ratings.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = TextAugmenter()
        # 预先进行文本清理
        self.cleaned_texts = [self.clean_text(str(text)) for text in self.texts]

    def clean_text(self, text):
        # 添加文本清理步骤
        text = re.sub(r'\s+', ' ', text)  # 规范化空白字符
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # 保留基本标点
        return text.strip()

    def augment_text(self, text):
        augmentation_ops = [
            (self.augmenter.synonym_replacement, {'n': 2}),
            (self.augmenter.random_deletion, {'p': 0.15}),
            (self.augmenter.random_swap, {'n': 2}),
            (self.augmenter.back_translation, {})
        ]
        # 随机选择多个增强方法
        num_augs = random.randint(1, 2)
        for _ in range(num_augs):
            op, params = random.choice(augmentation_ops)
            text = op(text, **params) if params else op(text)
        return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rating = self.ratings[idx]
        
        text = self.clean_text(text)
        if self.augment and random.random() < 0.4:  # 增加增强概率
            text = self.augment_text(text)
        
        # 增强提示信息
        sentiment = "negative" if rating <= 2 else "neutral" if rating <= 3 else "positive"
        prompt = f"Generate a {rating}-star {sentiment} product review:"
        
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=48,  # 增加prompt长度限制
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': prompt_encoding['input_ids'].squeeze(),
            'attention_mask': prompt_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }

class ReviewGenerator(nn.Module):
    def __init__(self, model_name="t5-base"):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        # 简单启用梯度检查点，不指定use_reentrant参数
        self.model.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False  # 禁用KV缓存以配合gradient checkpointing
        )
        return outputs


def prepare_data(batch_size=12):
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    data = pd.DataFrame(dataset["full"][:200000])
    
    data['text'] = data['text'].fillna("").astype(str)
    data = data[data['text'].str.len() > 20]
    data = data[data['text'].str.len() < 1000]
    
    samples_per_rating = min(data['rating'].value_counts())
    balanced_data = pd.concat([
        data[data['rating'] == r].sample(n=samples_per_rating, random_state=42)
        for r in range(1, 6)
    ])
    
    train_texts, val_texts, train_ratings, val_ratings = train_test_split(
        balanced_data['text'], balanced_data['rating'],
        test_size=0.1,
        random_state=42,
        stratify=balanced_data['rating']
    )
    
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512, legacy=False)
    
    train_dataset = ReviewGenerationDataset(train_texts, train_ratings, tokenizer, augment=True)
    val_dataset = ReviewGenerationDataset(val_texts, val_ratings, tokenizer, augment=False)
    
    # 减少worker数量，使用单进程来避免NLTK警告
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=0  # 使用单进程
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=0  # 使用单进程
    )
    
    return train_loader, val_loader, tokenizer

def train_generator(model, train_loader, val_loader, epochs=5, save_dir="generator_checkpoints", 
                   gradient_accumulation_steps=4):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("generative-output", exist_ok=True)
    
    # 修改优化器配置，添加梯度裁剪和学习率调整
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # 降低学习率
    num_training_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    num_warmup_steps = num_training_steps // 10  # 减少预热步数比例
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 3
    patience_counter = 0
    
    # 使用torch.cuda.amp.autocast()替代GradScaler
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_steps = 0
        running_loss = 0.0
        log_every = 50  # 降低loss打印频率
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                try:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
                        
                        # 添加梯度检查点以减少内存使用
                        with torch.set_grad_enabled(True):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                decoder_attention_mask=decoder_attention_mask
                            )
                            
                            loss = outputs.loss / gradient_accumulation_steps
                            
                            # 检查loss是否为NaN
                            if torch.isnan(loss):
                                print(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                                continue
                    
                    loss.backward()
                    
                    # 添加梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    running_loss += loss.item() * gradient_accumulation_steps
                    total_train_loss += loss.item() * gradient_accumulation_steps
                    epoch_steps += 1
                    
                    # 降低loss显示频率
                    if (batch_idx + 1) % log_every == 0:
                        avg_running_loss = running_loss / log_every
                        pbar.set_postfix({
                            'train_loss': f'{avg_running_loss:.4f}'
                        })
                        running_loss = 0.0
                    
                    pbar.update(1)
                
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    optimizer.zero_grad()
                    continue
        
        if len(train_loader) % gradient_accumulation_steps != 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        decoder_attention_mask = batch['decoder_attention_mask'].to(device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            decoder_attention_mask=decoder_attention_mask
                        )
                        
                        loss = outputs.loss
                        
                        # 检查验证loss是否为NaN
                        if not torch.isnan(loss):
                            total_val_loss += loss.item()
                            val_steps += 1
                
                except RuntimeError as e:
                    print(f"Error in validation batch: {str(e)}")
                    continue
        
        # 计算平均loss，确保不会出现除零错误
        avg_train_loss = total_train_loss / max(epoch_steps, 1)
        avg_val_loss = total_val_loss / max(val_steps, 1)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        # 保存检查点和早停逻辑
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"{save_dir}/best_model.pth")
            print("Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epoch + 2), train_losses, label='Training Loss')
        plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('generative-output/loss_curve.png')
        plt.close()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_review(model, tokenizer, rating, max_length=150, num_return_sequences=1):
    model.eval()
    sentiment = "negative" if rating <= 2 else "neutral" if rating <= 3 else "positive"
    prompt = f"Generate a {rating}-star {sentiment} product review:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=48,
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=3,  # 增加以避免重复
            top_k=50,
            top_p=0.92,
            temperature=0.85,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            length_penalty=1.2,  # 添加长度惩罚以产生更完整的句子
            early_stopping=True,  # 启用早停以避免生成未完成的句子
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # 确保生成到完整的句子结束
        )
    
    generated_texts = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # 确保句子完整性
        if not text.endswith(('.', '!', '?')):
            text += '.'
        generated_texts.append(text)
    
    return generated_texts[0] if num_return_sequences == 1 else generated_texts


def main():
    model = None  # 初始化为 None，用于异常处理检查
    print("Starting review generator training and evaluation...")
    
    try:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 准备数据
        print("\nPreparing datasets...")
        train_loader, val_loader, tokenizer = prepare_data()
        
        # 初始化模型
        print("\nInitializing model...")
        model = ReviewGenerator().to(device)
        
        # 训练模型
        print("\nStarting training...")
        train_generator(model, train_loader, val_loader)
        
        # 加载最佳模型进行评估
        print("\nLoading best model for generation...")
        model.load_state_dict(torch.load("generator_checkpoints/best_model.pth"))
        model.eval()
        
        # 生成示例评论
        print("\nGenerating sample reviews...")
        ratings = [1, 2, 3, 4, 5]
        num_samples = 3  # 每个评分生成多个样本
        
        with open('generative-output/generated_reviews.txt', 'w') as f:
            for rating in ratings:
                print(f"\n{'-'*50}")
                print(f"Generated {rating}-star reviews:")
                print(f"{'-'*50}")
                
                f.write(f"\n{'-'*50}\n")
                f.write(f"Generated {rating}-star reviews:\n")
                f.write(f"{'-'*50}\n")
                
                for i in range(num_samples):
                    review = generate_review(model, tokenizer, rating, max_length=200)
                    print(f"\nSample {i+1}:")
                    print(review)
                    
                    f.write(f"\nSample {i+1}:\n")
                    f.write(review + "\n")
        
        # 评估生成质量
        print("\nEvaluation complete. Generated reviews have been saved to 'generative-output/generated_reviews.txt'")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if model is not None:  # 检查模型是否已经创建
            save_path = "generator_checkpoints/interrupted_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model state saved to '{save_path}'")
    
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印完整的错误堆栈
        if model is not None:  # 检查模型是否已经创建
            save_path = "generator_checkpoints/error_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model state saved to '{save_path}'")
    
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()