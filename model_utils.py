"""Model utilities for saving and loading checkpoints."""

import os
import torch
from pathlib import Path


class CheckpointManager:
    """Manages model checkpoints - saving and loading."""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy, filename='checkpoint.pth'):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Training loss
            accuracy: Validation accuracy
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved to {filepath}')
    
    def load_checkpoint(self, model, optimizer, filename='checkpoint.pth'):
        """
        Load model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            filename: Checkpoint filename
            
        Returns:
            Dictionary with checkpoint metadata (epoch, loss, accuracy)
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f'Checkpoint not found at {filepath}')
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        metadata = {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss'],
            'accuracy': checkpoint['accuracy'],
        }
        
        print(f'Checkpoint loaded from {filepath}')
        return metadata
    
    def save_best_model(self, model, best_accuracy, filename='best_model.pth'):
        """
        Save the best model based on accuracy.
        
        Args:
            model: PyTorch model
            best_accuracy: Best validation accuracy achieved
            filename: Model filename
        """
        filepath = self.checkpoint_dir / filename
        state = {
            'model_state_dict': model.state_dict(),
            'best_accuracy': best_accuracy,
        }
        torch.save(state, filepath)
        print(f'Best model saved to {filepath} with accuracy {best_accuracy:.4f}')
    
    def load_best_model(self, model, filename='best_model.pth'):
        """
        Load the best saved model.
        
        Args:
            model: PyTorch model
            filename: Model filename
            
        Returns:
            Best accuracy
        """
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f'Model not found at {filepath}')
        
        state = torch.load(filepath)
        model.load_state_dict(state['model_state_dict'])
        best_accuracy = state['best_accuracy']
        
        print(f'Best model loaded from {filepath}')
        return best_accuracy
