from transformers import DistilBertModel, DistilBertConfig
import torch
import torch.nn as nn
import pandas as pd
import os
import re
import random
import nltk
from nltk.corpus import wordnet
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torch.amp import autocast  # updated import
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------- 強化 Tokenizer + 資料增強 ----------
class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.pad_id = 0
        self.unk_id = 1

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        return text.strip()

    

    def synonym_replace(self, text, prob=0.05):
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < prob and len(word) > 3:
                synsets = wordnet.synsets(word)
                lemmas = set(l.name().replace('_', ' ') for s in synsets for l in s.lemmas())
                lemmas.discard(word)
                if lemmas:
                    new_words.append(random.choice(list(lemmas)))
                    continue
            new_words.append(word)
        return ' '.join(new_words)

    def build_vocab(self, texts):
        vocab_set = set()
        for text in texts:
            clean = self.preprocess(text)
            vocab_set.update(clean.split())
        self.vocab = {word: i + 2 for i, word in enumerate(sorted(vocab_set))}
        self.vocab[self.pad_token] = self.pad_id
        self.vocab[self.unk_token] = self.unk_id

    def encode_batch(self, texts, max_length=128):
        input_ids, attention_mask = [], []
        for text in texts:
            text = self.synonym_replace(self.preprocess(text))
            tokens = [self.vocab.get(w, self.unk_id) for w in text.split()[:max_length]]
            pad_len = max_length - len(tokens)
            input_ids.append(tokens + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

# ---------- Dataset ----------
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

# ---------- 模型定義：加深 MLP，加入 LayerNorm 與 GELU ----------
class DistilBERT_Classifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        config = DistilBertConfig()
        config.max_position_embeddings = 128
        config.vocab_size = vocab_size
        self.bert = DistilBertModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (output.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)  # [CLS]
        return self.classifier(pooled)

# ---------- 資料讀取與編碼 ----------
def load_and_cache_data(tokenizer, max_length=128):
    print("Encoding and caching...")
    train_df = pd.read_csv('train_data.csv').dropna(subset=['text'])
    val_df = pd.read_csv('val_data.csv').dropna(subset=['text'])

    tokenizer.build_vocab(train_df['text'].tolist())
    print(f"✅ Vocabulary size: {len(tokenizer.vocab)}")
    if len(tokenizer.vocab) == 0:
        raise ValueError("Tokenizer vocabulary is empty!")

    train_encodings = tokenizer.encode_batch(train_df['text'].tolist(), max_length)
    val_encodings = tokenizer.encode_batch(val_df['text'].tolist(), max_length)

    train_labels = torch.tensor(train_df['label'].tolist(), dtype=torch.long)
    val_labels = torch.tensor(val_df['label'].tolist(), dtype=torch.long)

    return train_encodings, train_labels, val_encodings, val_labels

# ---------- Focal Loss ----------
class WeightedCELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)

# ---------- 訓練流程 ----------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = SimpleTokenizer()
    train_encodings, train_labels, val_encodings, val_labels = load_and_cache_data(tokenizer)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = DistilBERT_Classifier(vocab_size=len(tokenizer.vocab)).to(device)
    def get_layerwise_optimizer(model, base_lr=5e-5, weight_decay=0.01):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        layers = [model.bert.embeddings] + list(model.bert.transformer.layer) + [model.classifier]

        lr = base_lr
        decay_factor = 0.95

        for layer in layers:
            layer_params = list(layer.named_parameters())
            optimizer_grouped_parameters += [
                {'params': [p for n, p in layer_params if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay, 'lr': lr},
                {'params': [p for n, p in layer_params if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': lr}
            ]
            lr *= decay_factor

        return torch.optim.AdamW(optimizer_grouped_parameters)

    optimizer = get_layerwise_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    class_counts = train_labels.bincount().float().to(device)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    criterion = WeightedCELoss(class_weights)
    scaler = GradScaler()

    best_acc = 0
    patience, patience_counter = 5, 0
    logs = []

    for epoch in range(1, 21):
        model.train()
        train_preds, train_trues = [], []
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_preds += pred.cpu().tolist()
            train_trues += labels.cpu().tolist()
            loop.set_postfix(loss=loss.item())

        train_acc = accuracy_score(train_trues, train_preds)
        avg_loss = total_loss / len(train_loader)

        model.eval()
        preds, trues = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with autocast(device_type=device.type):
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                preds += pred.cpu().tolist()
                trues += labels.cpu().tolist()

        val_acc = accuracy_score(trues, preds)
        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch} Train Loss: {avg_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        logs.append([epoch, avg_loss, train_acc, avg_val_loss, val_acc])
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_distilbert.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\n✅ Best Validation Accuracy Achieved: {best_acc:.4f}")
    pd.DataFrame(logs, columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']) \
        .to_csv('training_log.csv', index=False)

if __name__ == '__main__':
    nltk.download('wordnet')
    train()
