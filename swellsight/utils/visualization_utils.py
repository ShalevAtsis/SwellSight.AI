"""Visualization utilities for SwellSight Wave Analysis Model."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def plot_predictions(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot prediction vs target comparisons.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract data
    pred_heights = [p['height_meters'] for p in predictions]
    true_heights = [t['height'] for t in targets]
    
    pred_types = [p['wave_type'] for p in predictions]
    true_types = [t['wave_type'] for t in targets]
    
    pred_directions = [p['direction'] for p in predictions]
    true_directions = [t['direction'] for t in targets]
    
    # Height regression scatter plot
    axes[0].scatter(true_heights, pred_heights, alpha=0.6)
    axes[0].plot([min(true_heights), max(true_heights)], 
                 [min(true_heights), max(true_heights)], 'r--', lw=2)
    axes[0].set_xlabel('True Height (m)')
    axes[0].set_ylabel('Predicted Height (m)')
    axes[0].set_title('Wave Height Predictions')
    axes[0].grid(True, alpha=0.3)
    
    # Wave type confusion matrix
    from sklearn.metrics import confusion_matrix
    wave_types = ['A_FRAME', 'CLOSEOUT', 'BEACH_BREAK', 'POINT_BREAK']
    cm_type = confusion_matrix(true_types, pred_types, labels=wave_types)
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Blues', 
                xticklabels=wave_types, yticklabels=wave_types, ax=axes[1])
    axes[1].set_xlabel('Predicted Type')
    axes[1].set_ylabel('True Type')
    axes[1].set_title('Wave Type Confusion Matrix')
    
    # Direction confusion matrix
    directions = ['LEFT', 'RIGHT', 'BOTH']
    cm_direction = confusion_matrix(true_directions, pred_directions, labels=directions)
    sns.heatmap(cm_direction, annot=True, fmt='d', cmap='Greens',
                xticklabels=directions, yticklabels=directions, ax=axes[2])
    axes[2].set_xlabel('Predicted Direction')
    axes[2].set_ylabel('True Direction')
    axes[2].set_title('Direction Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training loss values per epoch
        val_losses: Validation loss values per epoch
        train_metrics: Training metrics per epoch
        val_metrics: Validation metrics per epoch
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height MAE
    if 'height_mae' in train_metrics:
        axes[0, 1].plot(epochs, train_metrics['height_mae'], 'b-', label='Training MAE')
        axes[0, 1].plot(epochs, val_metrics['height_mae'], 'r-', label='Validation MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE (meters)')
        axes[0, 1].set_title('Height Prediction MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Wave type accuracy
    if 'wave_type_accuracy' in train_metrics:
        axes[1, 0].plot(epochs, train_metrics['wave_type_accuracy'], 'b-', label='Training Accuracy')
        axes[1, 0].plot(epochs, val_metrics['wave_type_accuracy'], 'r-', label='Validation Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Wave Type Classification Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Direction accuracy
    if 'direction_accuracy' in train_metrics:
        axes[1, 1].plot(epochs, train_metrics['direction_accuracy'], 'b-', label='Training Accuracy')
        axes[1, 1].plot(epochs, val_metrics['direction_accuracy'], 'r-', label='Validation Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Direction Classification Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_height_distribution(
    heights: List[float],
    title: str = "Wave Height Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot distribution of wave heights.
    
    Args:
        heights: List of wave heights in meters
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(heights, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(heights), color='red', linestyle='--', 
                label=f'Mean: {np.mean(heights):.2f}m')
    plt.axvline(np.median(heights), color='green', linestyle='--', 
                label=f'Median: {np.median(heights):.2f}m')
    
    plt.xlabel('Wave Height (meters)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_distribution(
    classes: List[str],
    class_names: List[str],
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Plot distribution of classification labels.
    
    Args:
        classes: List of class labels
        class_names: List of all possible class names
        title: Plot title
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    from collections import Counter
    
    plt.figure(figsize=(10, 6))
    
    counts = Counter(classes)
    class_counts = [counts.get(name, 0) for name in class_names]
    
    bars = plt.bar(class_names, class_counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, class_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()