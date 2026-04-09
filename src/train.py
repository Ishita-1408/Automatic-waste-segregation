"""
Automatic Waste Segregation - Training Script
==============================================
Trains a CNN model on the Garbage Classification dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/mostafaabla/garbage-classification

Classes are automatically detected from dataset folders.
Current dataset contains 12 waste classes.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── Config ──────────────────────────────────────────────────────────────────

CLASS_NAMES = None
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data Transforms ─────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─── Model ───────────────────────────────────────────────────────────────────


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """EfficientNet-B0 for waste classification (no internet download)."""
    model = models.efficientnet_b0(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model

# ─── Training Loop ────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

# ─── Plotting ────────────────────────────────────────────────────────────────


def plot_history(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_acc"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(labels, preds, class_names, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

# ─── Main ────────────────────────────────────────────────────────────────────


def main(data_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Data  : {data_dir}")

    # Dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    class_names = full_dataset.classes
    n = len(full_dataset)
    n_val = int(0.15 * n)
    n_test = int(0.10 * n)
    n_train = n - n_val - n_test

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    test_ds.dataset = datasets.ImageFolder(data_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    print(f"[INFO] Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print("[INFO] Building model...")

    # Model
    model = build_model(len(class_names)).to(DEVICE)
    print("[INFO] Model built successfully.")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print("[INFO] Starting training loop...")
        # Unfreeze backbone after epoch 5
        if epoch == 6:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=LR * 0.1, weight_decay=1e-4)

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f})")

    # Load best & evaluate on test
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pth")))
    _, test_acc, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n[RESULT] Test Accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=class_names))

    # Save artefacts
    plot_history(history, output_dir)
    plot_confusion_matrix(labels, preds, class_names, output_dir)

    meta = {
        "model": "EfficientNet-B0",
        "num_classes": len(class_names),
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "trained_at": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Artefacts saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Waste Segregation Model")
    parser.add_argument("--data", default="data/garbage_classification",
                        help="Path to dataset root")
    parser.add_argument("--output", default="models/",
                        help="Where to save model & plots")
    args = parser.parse_args()
    main(args.data, args.output)
