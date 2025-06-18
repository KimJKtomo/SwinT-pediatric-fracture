import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score, confusion_matrix
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm

# === Custom Mixed Loss ===
class MixedFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, bce_ratio=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_ratio = bce_ratio

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return (1 - self.bce_ratio) * focal_loss.mean() + self.bce_ratio * bce_loss.mean()

# === Evaluation for multiple thresholds ===
def evaluate_thresholds(y_true, y_probs):
    thresholds = [round(x, 2) for x in np.linspace(0.3, 0.7, 21)]
    for th in thresholds:
        y_pred = (y_probs > th).astype(int)
        r0 = recall_score(y_true, y_pred, pos_label=0)
        r1 = recall_score(y_true, y_pred, pos_label=1)
        f1 = f1_score(y_true, y_pred)
        print(f"[th={th:.2f}] Recall_0: {r0:.4f} | Recall_1: {r1:.4f} | F1: {f1:.4f}")

# === Auto Alpha Calculation ===
def compute_alpha(train_labels):
    counter = Counter(train_labels)
    N0 = counter[0]
    N1 = counter[1]
    alpha = N1 / (N0 + N1)
    print(f"[INFO] Auto-calculated alpha for FocalLoss: {alpha:.4f}")
    return alpha

# === Dummy Model ===
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 1)  # dummy for grayscale or RGB
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# === Dummy Dataset ===
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.images = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 2, (size,))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.size

# === Training Script ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = DummyDataset(8000)
    val_dataset = DummyDataset(2000)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Alpha 계산
    train_labels = [label for _, label in train_dataset]
    alpha = compute_alpha(train_labels)

    # Model & Optimizer
    model = DummyModel().to(device)
    criterion = MixedFocalBCELoss(alpha=alpha, gamma=2.0, bce_ratio=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_f1 = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y.numpy())

        y_probs = np.array(all_preds)
        y_true = np.array(all_labels)

        print(f"\n[Epoch {epoch}] Threshold sweep on validation:")
        evaluate_thresholds(y_true, y_probs)

train()
