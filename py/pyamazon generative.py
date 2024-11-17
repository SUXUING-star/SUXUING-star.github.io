# 导入所需库
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    get_linear_schedule_with_warmup
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

# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import wordnet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()
# 数据增强方法
class TextAugmenter:
    @staticmethod
    def synonym_replacement(text, n=1):
        words = word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalnum()]))
        n = min(n, len(random_word_list))
        
        for _ in range(n):
            random_word = random.choice(random_word_list)
            synonyms = []
            for syn in wordnet.synsets(random_word):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name())
            if len(synonyms) > 0:
                synonym = random.choice(list(set(synonyms)))
                random_idx = random.randint(0, len(words) - 1)
                new_words[random_idx] = synonym
        
        return ' '.join(new_words)

    @staticmethod
    def random_deletion(text, p=0.1):
        words = word_tokenize(text)
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words)-1)
            new_words.append(words[rand_int])
            
        return ' '.join(new_words)

    @staticmethod
    def random_swap(text, n=1):
        words = word_tokenize(text)
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) >= 2:
                idx1, idx2 = random.sample(range(len(new_words)), 2)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        return ' '.join(new_words)

class ReviewGenerationDataset(Dataset):
    def __init__(self, texts, ratings, tokenizer, max_len=128, augment=False):
        # 将Series转换为list以避免索引问题
        self.texts = texts.tolist()
        self.ratings = ratings.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.augmenter = TextAugmenter()

    def __len__(self):
        return len(self.texts)

    def augment_text(self, text):
        augmentation_ops = [
            (self.augmenter.synonym_replacement, {'n': 1}),
            (self.augmenter.random_deletion, {'p': 0.1}),
            (self.augmenter.random_swap, {'n': 1})
        ]
        op, params = random.choice(augmentation_ops)
        return op(text, **params)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        rating = self.ratings[idx]
        
        if self.augment and random.random() < 0.3:
            text = self.augment_text(text)
        
        prompt = f"Generate a {rating}-star review:"
        
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=32,
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
        # 启用梯度检查点以减少内存使用
        self.model.gradient_checkpointing_enable()
        
    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs


def prepare_data(batch_size=8):  # 减小batch size以降低内存使用
    # 加载数据集
    dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
    data = pd.DataFrame(dataset["full"][:200000])
    
    # 数据清洗
    data['text'] = data['text'].fillna("").astype(str)
    data = data[data['text'].str.len() > 10]
    data = data.reset_index(drop=True)
    
    # 划分训练集和验证集
    train_texts, val_texts, train_ratings, val_ratings = train_test_split(
        data['text'], data['rating'],
        test_size=0.1,
        random_state=42
    )
    
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)  # 减小max_length
    
    train_dataset = ReviewGenerationDataset(train_texts, train_ratings, tokenizer, augment=True)
    val_dataset = ReviewGenerationDataset(val_texts, val_ratings, tokenizer, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        pin_memory=True  # 启用pin_memory加速数据传输
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        drop_last=True,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer


def train_generator(model, train_loader, val_loader, epochs=5, save_dir="generator_checkpoints", 
                   gradient_accumulation_steps=4):
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("generative-output", exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_training_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # 将数据移动到GPU
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
                
                loss = outputs.loss / gradient_accumulation_steps
                total_train_loss += loss.item() * gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # 清理缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                pbar.update(1)
                pbar.set_postfix({
                    'train_loss': f'{loss.item() * gradient_accumulation_steps:.4f}'
                })
        
        # 处理最后一个不完整的累积步
        if len(train_loader) % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # 评估阶段
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
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
                
                total_val_loss += outputs.loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average validation loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
            print("Saved new best model!")
        
        # 绘制并保存loss曲线
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
        
        # 强制进行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_review(model, tokenizer, rating, max_length=150):
    model.eval()
    prompt = f"Generate a {rating}-star review:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=32,
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    # 清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    train_loader, val_loader, tokenizer = prepare_data()
    model = ReviewGenerator().to(device)
    train_generator(model, train_loader, val_loader)
    
    # 生成示例评论
    model.load_state_dict(torch.load("generator_checkpoints/best_model.pth"))
    
    for rating in [1, 3, 5]:
        print(f"\nGenerated {rating}-star review:")
        review = generate_review(model, tokenizer, rating)
        print(review)

if __name__ == "__main__":
    main()