import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as F
from tqdm import tqdm
from PIL import Image, ImageFilter


class SyntheticForgeryDataset(Dataset):
    """
    Wraps CIFAR-10 and turns it into a binary classification dataset:
    - Label 0: authentic (original image)
    - Label 1: forged (synthetically tampered image)
    """

    def __init__(self, root: str, train: bool = True, download: bool = True):
        self.base_dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # we will apply transforms manually
        )

        # Transforms for authentic images (resize + normalize)
        self.auth_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Transforms for forged images (tampering)
        self.forgery_augment = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.random_erasing = transforms.RandomErasing(
            p=0.7,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value=0
        )

    def __len__(self) -> int:
        # We will create one authentic and one forged sample per original image
        return len(self.base_dataset) * 2

    def _create_forgery(self, img):
        # Basic tampering pipeline: jitter, blur, random erase
        img = self.forgery_augment(img)
        # Convert to tensor for RandomErasing
        img_tensor = transforms.ToTensor()(img)

        # Apply random erasing (simulating erased / hidden region)
        img_tensor = self.random_erasing(img_tensor)

        # Convert back to PIL for consistency with downstream transforms
        img = F.to_pil_image(img_tensor)

        # Optional: small Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Final tensor + normalization
        img_tensor = self.to_tensor_norm(img)
        return img_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = idx // 2
        is_forged = (idx % 2 == 1)

        img, _ = self.base_dataset[base_idx]

        if is_forged:
            img_tensor = self._create_forgery(img)
            label = 1
        else:
            img_tensor = self.auth_transform(img)
            label = 0

        return img_tensor, label


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(images)

        # BCEWithLogitsLoss expects shape [batch_size, 1] for binary
        logits = outputs[:, 1]  # take logit for class "1" (forged)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(logits) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val", leave=False):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            logits = outputs[:, 1]
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters – you can tweak these
    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    val_split = 0.2
    data_root = "./data"

    # Dataset
    full_train = SyntheticForgeryDataset(root=data_root, train=True, download=True)

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, ImageFilternum_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, ImageFilternum_workers=0)

    # Model, loss, optimizer
    model = build_model(num_classes=2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("checkpoints", "best_model.pt"))
            print(f"Saved new best model with val acc = {best_val_acc:.4f}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
