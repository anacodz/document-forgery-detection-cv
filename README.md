# Document Forgery Detection (Synthetic) – PyTorch

This project implements a simple deep learning pipeline in PyTorch that simulates **document forgery detection** as a binary image classification task. The goal is to distinguish between "authentic" and "forged" images using transfer learning and synthetic tampering, reflecting real-world fraud prevention scenarios.

Although real document datasets are often private, this project uses the public CIFAR-10 dataset and creates a synthetic "forgery" class using data augmentation (blur, cutout, random overlays). This demonstrates the overall **end-to-end machine learning lifecycle**: problem framing, data processing, model training, evaluation, and error analysis.

## Problem formulation

- **Input**: Color images (CIFAR-10 sample images resized to 224×224).
- **Classes**:
  - Class 0 – "Authentic": original images.
  - Class 1 – "Forged": images with synthetic tampering such as blur, random erasing, and noise overlays.
- **Task**: Binary classification (authentic vs forged).

This setup mimics real-world **document/image forgery detection** where tampered regions can include blurred sections, erased areas, or noisy overlays. It aligns with applications such as **ID verification and fraud prevention**.

## Approach

- Framework: **PyTorch** and **torchvision** for datasets, transforms, and pretrained models.
- Dataset: CIFAR-10 downloaded via torchvision (`torchvision.datasets.CIFAR10`).
- Synthetic forgery generation:
  - Gaussian blur.
  - RandomErasing.
  - Random horizontal flip and color jitter.
- Model:
  - Pretrained **ResNet18** from `torchvision.models` with the final layer modified for 2 output classes (authentic vs forged).
- Training:
  - Binary cross‑entropy with logits (`BCEWithLogitsLoss`).
  - Adam optimizer.
  - Simple training loop with accuracy metrics on validation set.
- Evaluation:
  - Validation accuracy.
  - Example predictions and some misclassified samples (can be added in a notebook if desired).

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_forgery_classifier.py
```

This will:

- Download CIFAR-10 if not already present.
- Create a synthetic binary dataset (authentic vs forged).
- Train a ResNet18-based classifier for a few epochs.
- Print training and validation accuracy per epoch.

You can adjust hyperparameters (epochs, batch size, learning rate) inside `train_forgery_classifier.py`.

## Future improvements

- Replace CIFAR-10 with a real document / ID / receipt dataset.
- Use object detection or segmentation models to localize tampered regions.
- Integrate OCR and consistency checks between visual content and text.
- Add experiment tracking (TensorBoard/W&B) and more advanced augmentations.

## Relevance

Even though the dataset is synthetic, this project demonstrates:

- Deep learning for image-based fraud/forgery detection.
- Hands-on experience with PyTorch, transfer learning, and image augmentations.
- End-to-end ML workflow aligned with real-world identity verification and document analysis.
