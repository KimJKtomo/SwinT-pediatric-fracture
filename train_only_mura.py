# train_only_mura.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier
from load_new_dxmodule_0620_ao import load_mura

class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 1
    batch_size = 8
    lr = 1e-5
    num_epochs = 50
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    threshold = 0.5
    patience = 10
    run_name = "MURA_Train_ImageLevel"
    model_path = "swinT_muraonly_best.pt"

def stratified_split(df):
    train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    frac = test_df[test_df['label'] == 1]
    norm = test_df[test_df['label'] == 0]
    n = min(len(frac), len(norm))
    test_bal = pd.concat([frac.sample(n, random_state=42), norm.sample(n, random_state=42)])
    test_bal = test_bal.sample(frac=1, random_state=42).reset_index(drop=True)
    train_val_df = df[~df.index.isin(test_bal.index)]
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, stratify=train_val_df['label'], random_state=42)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_bal

def evaluate(model, loader, args, save_roc=False, name="val"):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            out = model(x)
            prob = torch.sigmoid(out).cpu().numpy().flatten()
            probs += prob.tolist()
            labels += y.numpy().tolist()
    bin_pred = [1 if p > args.threshold else 0 for p in probs]
    acc = accuracy_score(labels, bin_pred)
    f1 = f1_score(labels, bin_pred)
    cm = confusion_matrix(labels, bin_pred)

    if save_roc:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title(f"ROC Curve ({name})")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"roc_curve_{name}.png")
        mlflow.log_artifact(f"roc_curve_{name}.png")
    return acc, f1, cm

def train():
    args = Args()
    mlflow.start_run(run_name=args.run_name)
    mlflow.log_params(vars(args))

    df = load_mura()
    print(f"\n📊 Dataset Summary:")
    print(f"Total Image Samples: {len(df)}")
    print(f"Fracture: {df['label'].sum()} | Normal: {(df['label'] == 0).sum()}")

    train_df, val_df, test_df = stratified_split(df)

    tf = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    train_labels = train_df['label'].tolist()
    class_weights = 1.0 / torch.tensor([train_labels.count(0), train_labels.count(1)]).float()
    weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(UnifiedFractureDataset(train_df, tf), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(UnifiedFractureDataset(val_df, tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(UnifiedFractureDataset(test_df, tf), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SwinTImageClassifier(args).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1, counter = 0, 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss, preds, labels = 0, [], []
        for x, y in tqdm(train_loader, desc=f"[Epoch {epoch}] Train"):
            x, y = x.to(args.device), y.float().to(args.device)
            out = model(x)
            loss = loss_fn(out, y.view(-1, 1))  # ✅ 핵심 수정
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            preds += torch.sigmoid(out).cpu().detach().numpy().flatten().tolist()
            labels += y.cpu().numpy().flatten().tolist()
        p_bin = [1 if p > args.threshold else 0 for p in preds]
        tr_f1 = f1_score(labels, p_bin)
        tr_acc = accuracy_score(labels, p_bin)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f} | Acc: {tr_acc:.4f} | F1: {tr_f1:.4f}")
        mlflow.log_metric("train_f1", tr_f1, step=epoch)
        mlflow.log_metric("train_acc", tr_acc, step=epoch)

        val_acc, val_f1, val_cm = evaluate(model, val_loader, args, save_roc=(epoch==args.num_epochs-1), name="val")
        print(f"[Epoch {epoch}] Val Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}\nConfusion Matrix:\n{val_cm}")
        mlflow.log_metric("val_f1", val_f1, step=epoch)
        mlflow.log_metric("val_acc", val_acc, step=epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), args.model_path)
            mlflow.log_artifact(args.model_path)
            print("✅ Best model saved.")
        else:
            counter += 1
            if counter >= args.patience:
                print("⏹️ Early stopping.")
                break

    # 🧪 Test
    print("\n🧪 Final Test Evaluation...")
    model.load_state_dict(torch.load(args.model_path))
    acc, f1, cm = evaluate(model, test_loader, args, save_roc=True, name="test")
    print(f"Test Accuracy: {acc:.4f} | F1: {f1:.4f}\nConfusion Matrix:\n{cm}")
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_acc", acc)
    mlflow.end_run()

if __name__ == '__main__':
    train()
