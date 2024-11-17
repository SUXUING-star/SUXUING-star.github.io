class EnhancedBERTClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        hidden_size = self.bert.config.hidden_size
        
        # 特征融合层
        self.feature_fusion = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(4)  # 使用最后4层的特征
        ])
        
        # 多头自注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1
        )
        
        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, input_ids, attention_mask):
        # 1. 获取BERT所有层的输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 2. 多层特征融合
        last_hidden_states = outputs.hidden_states[-4:]
        fused_features = []
        for i, hidden_state in enumerate(last_hidden_states):
            fused_features.append(self.feature_fusion[i](hidden_state))
        
        # 3. 特征加权融合
        fused_output = torch.stack(fused_features).mean(dim=0)
        
        # 4. 应用多头注意力
        attn_output, _ = self.multihead_attn(
            fused_output.transpose(0, 1),
            fused_output.transpose(0, 1),
            fused_output.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = attn_output.transpose(0, 1)
        
        # 5. 特征增强
        enhanced_features = self.feature_enhancement(attn_output)
        
        # 6. 获取[CLS]标记的输出并分类
        cls_output = enhanced_features[:, 0]
        return self.classifier(cls_output)

def train_model(model, train_loader, val_loader, epochs=10):
    # 使用标签平衡的损失函数
    weights = torch.tensor([1.2, 0.9, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 优化器设置
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 1e-5},
        {'params': model.feature_fusion.parameters(), 'lr': 2e-4},
        {'params': model.multihead_attn.parameters(), 'lr': 2e-4},
        {'params': model.feature_enhancement.parameters(), 'lr': 2e-4},
        {'params': model.classifier.parameters(), 'lr': 2e-4}
    ], weight_decay=0.01)
    
    # 学习率调度
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        predictions, true_labels = [], []
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
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
                
                pbar.update(1)
                
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
        
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Average train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
        print(f'Average val loss: {avg_val_loss:.4f}, accuracy: {val_acc:.4f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
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
    plt.savefig("figures/loss_curve.png")
    plt.close()
    
    return train_losses, val_losses, train_accs, val_accs