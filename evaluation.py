"""Evaluation script for Document Forgery Detection model.

Computes metrics like accuracy, precision, recall, F1-score, and confusion matrix.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test set and return metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': confusion_matrix(all_labels, all_preds),
    }
    
    return metrics, all_labels, all_preds, all_probs


def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Authentic', 'Forged'],
                yticklabels=['Authentic', 'Forged'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'Confusion matrix saved to {save_path}')
    plt.close()


def plot_roc_curve(all_labels, all_probs, save_path='roc_curve.png'):
    """
    Plot and save ROC curve.
    
    Args:
        all_labels: True labels
        all_probs: Predicted probabilities
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc = roc_auc_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'ROC curve saved to {save_path}')
    plt.close()


if __name__ == '__main__':
    print('Evaluation script loaded. Use evaluate_model() function to evaluate your model.')
