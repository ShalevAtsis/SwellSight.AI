"""Training utilities for SwellSight Wave Analysis Model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import time
from datetime import datetime
import numpy as np

from ..models import WaveAnalysisModel, MultiTaskLoss
from ..config import TrainingConfig, DataConfig
from ..data import DatasetManager
from ..utils.checkpoint_utils import save_checkpoint, load_checkpoint
from ..utils.device_utils import get_device

# Set up logging
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Training manager for the wave analysis model.
    
    Implements multi-task training loop with checkpointing, validation metrics tracking,
    and early stopping as specified in requirements 3.5 and 8.2.
    """
    
    def __init__(
        self, 
        model: WaveAnalysisModel, 
        training_config: TrainingConfig,
        data_config: DataConfig,
        output_dir: Path,
        dataset_manager: Optional[DatasetManager] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Wave analysis model to train
            training_config: Training configuration
            data_config: Data configuration
            output_dir: Output directory for checkpoints and logs
            dataset_manager: Optional dataset manager (will create if not provided)
        """
        self.model = model
        self.training_config = training_config
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device
        self.device = get_device()
        self.model = self.model.to(self.device)
        
        # Initialize dataset manager
        if dataset_manager is None:
            self.dataset_manager = DatasetManager(
                data_path=data_config.data_path,
                config=data_config.to_dict()
            )
        else:
            self.dataset_manager = dataset_manager
        
        # Initialize multi-task loss function
        self.criterion = MultiTaskLoss().to(self.device)
        
        # Set initial loss weights if specified in config
        if hasattr(training_config, 'initial_height_weight'):
            self.criterion.height_weight.data = torch.tensor(training_config.initial_height_weight)
        if hasattr(training_config, 'initial_type_weight'):
            self.criterion.wave_type_weight.data = torch.tensor(training_config.initial_type_weight)
        if hasattr(training_config, 'initial_direction_weight'):
            self.criterion.direction_weight.data = torch.tensor(training_config.initial_direction_weight)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=training_config.early_stopping_patience,
            min_delta=training_config.early_stopping_min_delta
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_height_mae': [],
            'val_height_mae': [],
            'train_type_acc': [],
            'val_type_acc': [],
            'train_direction_acc': [],
            'val_direction_acc': []
        }
        
        logger.info(f"Initialized Trainer with device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.training_config.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config.optimizer}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        if self.training_config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config.num_epochs,
                eta_min=self.training_config.learning_rate * 0.01
            )
        elif self.training_config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.training_config.num_epochs // 3,
                gamma=0.1
            )
        elif self.training_config.scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif self.training_config.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {self.training_config.scheduler}")
    
    def _compute_metrics(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute training/validation metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Height regression metrics (MAE)
        height_pred = predictions['height'].squeeze()
        height_target = targets['height']
        height_mae = torch.mean(torch.abs(height_pred - height_target)).item()
        metrics['height_mae'] = height_mae
        
        # Wave type classification accuracy
        type_pred = torch.argmax(predictions['wave_type'], dim=1)
        type_target = targets['wave_type']
        type_acc = (type_pred == type_target).float().mean().item()
        metrics['type_acc'] = type_acc
        
        # Direction classification accuracy
        direction_pred = torch.argmax(predictions['direction'], dim=1)
        direction_target = targets['direction']
        direction_acc = (direction_pred == direction_target).float().mean().item()
        metrics['direction_acc'] = direction_acc
        
        return metrics
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0.0
        total_metrics = {'height_mae': 0.0, 'type_acc': 0.0, 'direction_acc': 0.0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            images = batch['image'].to(self.device)
            targets = {
                'height': batch['height'].to(self.device),
                'wave_type': batch['wave_type'].to(self.device),
                'direction': batch['direction'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Compute loss
            loss_dict = self.criterion(predictions, targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Compute metrics
            batch_metrics = self._compute_metrics(predictions, targets)
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in batch_metrics.items():
                total_metrics[key] += value
            
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average metrics over all batches
        avg_metrics = {
            'loss': total_loss / num_batches,
            'height_mae': total_metrics['height_mae'] / num_batches,
            'type_acc': total_metrics['type_acc'] / num_batches,
            'direction_acc': total_metrics['direction_acc'] / num_batches
        }
        
        return avg_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics for the epoch
        """
        self.model.eval()
        
        total_loss = 0.0
        total_metrics = {'height_mae': 0.0, 'type_acc': 0.0, 'direction_acc': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                images = batch['image'].to(self.device)
                targets = {
                    'height': batch['height'].to(self.device),
                    'wave_type': batch['wave_type'].to(self.device),
                    'direction': batch['direction'].to(self.device)
                }
                
                # Forward pass
                predictions = self.model(images)
                
                # Compute loss
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss']
                
                # Compute metrics
                batch_metrics = self._compute_metrics(predictions, targets)
                
                # Accumulate metrics
                total_loss += loss.item()
                for key, value in batch_metrics.items():
                    total_metrics[key] += value
                
                num_batches += 1
        
        # Average metrics over all batches
        avg_metrics = {
            'loss': total_loss / num_batches,
            'height_mae': total_metrics['height_mae'] / num_batches,
            'type_acc': total_metrics['type_acc'] / num_batches,
            'direction_acc': total_metrics['direction_acc'] / num_batches
        }
        
        return avg_metrics
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """
        Save model checkpoint with metadata.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        
        # Prepare metadata
        metadata = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_config': self.training_config.to_dict(),
            'data_config': self.data_config.to_dict(),
            'model_config': self.model.config.to_dict(),
            'dataset_info': self.dataset_manager.get_dataset_info(),
            'training_time': datetime.now().isoformat(),
            'device': str(self.device)
        }
        
        # Save checkpoint
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            loss=val_metrics['loss'],
            metrics=val_metrics,
            config=self.training_config.to_dict(),
            filepath=checkpoint_path,
            metadata=metadata
        )
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model if this is the best validation loss
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            best_model_path = self.output_dir / "best_model.pth"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics=val_metrics,
                config=self.training_config.to_dict(),
                filepath=best_model_path,
                metadata=metadata
            )
            logger.info(f"Saved best model: {best_model_path}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model with multi-task optimization.
        
        Implements training loop with checkpointing every 10 epochs,
        validation metrics tracking, and early stopping as required.
        
        Returns:
            Training results and final metrics
        """
        logger.info("Starting training...")
        logger.info(f"Training for {self.training_config.num_epochs} epochs")
        logger.info(f"Checkpoint frequency: every {self.training_config.checkpoint_frequency} epochs")
        
        # Get data loaders
        train_loader = self.dataset_manager.get_train_loader(self.training_config.batch_size)
        val_loader = self.dataset_manager.get_validation_loader(self.training_config.batch_size)
        
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        start_time = time.time()
        
        for epoch in range(1, self.training_config.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase (every validation_frequency epochs)
            if epoch % self.training_config.validation_frequency == 0:
                val_metrics = self._validate_epoch(val_loader)
            else:
                val_metrics = {'loss': float('inf'), 'height_mae': 0.0, 'type_acc': 0.0, 'direction_acc': 0.0}
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Record training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_height_mae'].append(train_metrics['height_mae'])
            self.training_history['val_height_mae'].append(val_metrics['height_mae'])
            self.training_history['train_type_acc'].append(train_metrics['type_acc'])
            self.training_history['val_type_acc'].append(val_metrics['type_acc'])
            self.training_history['train_direction_acc'].append(train_metrics['direction_acc'])
            self.training_history['val_direction_acc'].append(val_metrics['direction_acc'])
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"Epoch {epoch:3d}/{self.training_config.num_epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Height MAE: {val_metrics['height_mae']:.3f} | "
                f"Type Acc: {val_metrics['type_acc']:.3f} | "
                f"Dir Acc: {val_metrics['direction_acc']:.3f}"
            )
            
            # Save checkpoint every checkpoint_frequency epochs
            if epoch % self.training_config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Check early stopping
            if val_metrics['loss'] != float('inf'):
                if self.early_stopping(val_metrics['loss']):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Save final checkpoint
        final_train_metrics = self._train_epoch(train_loader)
        final_val_metrics = self._validate_epoch(val_loader)
        self._save_checkpoint(self.current_epoch, final_train_metrics, final_val_metrics)
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        return {
            'status': 'completed',
            'epochs_trained': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_metrics': final_train_metrics,
            'final_val_metrics': final_val_metrics,
            'training_history': self.training_history,
            'total_training_time': total_time,
            'early_stopped': self.early_stopping.early_stop
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on validation set.
        
        Returns:
            Validation metrics
        """
        val_loader = self.dataset_manager.get_validation_loader(self.training_config.batch_size)
        return self._validate_epoch(val_loader)
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint_info = load_checkpoint(
            filepath=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device
        )
        
        self.current_epoch = checkpoint_info['epoch']
        self.best_val_loss = checkpoint_info.get('loss', float('inf'))
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint_info