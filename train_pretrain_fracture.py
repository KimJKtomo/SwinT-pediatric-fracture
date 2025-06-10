import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torchvision import transforms
from tqdm import tqdm

from load_new_dxmodule_0605 import get_combined_dataset
from unified_dataset_0605 import UnifiedFractureDataset
from SwinT_ImageOnly_Classifier import SwinTImageClassifier


class Args:
    model_name = "swin_large_patch4_window12_384_in22k"
    pretrained = True
    num_classes = 1
    batch_size = 8
    lr = 1e-4
    num_epochs = 5
    num_workers = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    args = Args()

    # 1. Dataset Load
    df = get_combined_dataset(image_only=True, fracture_only=True)
    df = df[df['split'].isin(['train', 'val'])]
    df = df[df['label'].isin([0, 1])].reset_index(drop=True)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)

    # 2. Transform
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_ds = UnifiedFractureDataset(train_df, transform=transform)
    val_ds = UnifiedFractureDataset(val_df, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Model + Optimizer
    model = SwinTImageClassifier(args).to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss, all_preds, all_targets = 0, [], []

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch}] Training"):
            images = images.to(args.device)
            labels = labels.float().to(args.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds += torch.sigmoid(outputs).detach().cpu().numpy().flatten().tolist()
            all_targets += labels.cpu().numpy().flatten().tolist()

        # Metrics
        bin_preds = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_targets, bin_preds)
        f1 = f1_score(all_targets, bin_preds)
        cm = confusion_matrix(all_targets, bin_preds)

        print(f"\n[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}")
        print(f"[Epoch {epoch}] Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
        print(f"[Epoch {epoch}] Confusion Matrix:\n{cm}")

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"[Epoch {epoch}] Validation"):
                images = images.to(args.device)
                labels = labels.float().to(args.device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                val_preds += probs.cpu().numpy().flatten().tolist()
                val_trues += labels.cpu().numpy().flatten().tolist()

        val_bin_preds = [1 if p > 0.5 else 0 for p in val_preds]
        val_f1 = f1_score(val_trues, val_bin_preds)
        val_cm = confusion_matrix(val_trues, val_bin_preds)
        val_acc = accuracy_score(val_trues, val_bin_preds)

        print(f"[Epoch {epoch}] Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        print(f"[Epoch {epoch}] Val Confusion Matrix:\n{val_cm}")

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "./swinT_pretrained_fx_best.pt")
            print("✅ Best model saved.")


if __name__ == "__main__":
    train()
