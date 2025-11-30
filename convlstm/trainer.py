"""Training pipeline for ConvLSTM weather prediction model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import time
import sys

from convlstm.model import ConvLSTMUNet, WeightedPrecipitationLoss
from convlstm.model_deep import DeepConvLSTMUNet
from convlstm.config import ConvLSTMConfig
from convlstm.data import ConvLSTMDataset, RegionConfig, ConvLSTMNormalizer
from convlstm.masked_loss import MaskedMSELoss, CombinedMaskedLoss


# Module-level logger
logger = logging.getLogger(__name__)


def save_model_checkpoint(
    model: Union[ConvLSTMUNet, 'DeepConvLSTMUNet'],
    filepath: Union[str, Path],
    config: Optional[ConvLSTMConfig] = None,
    region_config: Optional[RegionConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    epoch: int = 0,
    step: int = 0,
    best_val_loss: float = float('inf'),
    is_best: bool = False,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save model checkpoint with comprehensive metadata.
    
    This is a standalone function for saving model checkpoints that can be used
    independently of the ConvLSTMTrainer class. It saves model parameters along
    with optional training state and configuration.
    
    Args:
        model: ConvLSTMUNet or DeepConvLSTMUNet model to save
        filepath: Path to save checkpoint file
        config: Optional ConvLSTMConfig with model hyperparameters
        region_config: Optional RegionConfig with spatial boundaries
        optimizer: Optional optimizer state to save
        scheduler: Optional learning rate scheduler state to save
        epoch: Current training epoch (default: 0)
        step: Current training step (default: 0)
        best_val_loss: Best validation loss achieved (default: inf)
        is_best: Whether this is the best model so far (default: False)
        additional_metadata: Optional dictionary of additional metadata to save
    
    Example:
        >>> model = ConvLSTMUNet(input_channels=56, hidden_channels=[32, 64])
        >>> save_model_checkpoint(
        ...     model=model,
        ...     filepath='checkpoints/model.pt',
        ...     config=config,
        ...     epoch=10,
        ...     best_val_loss=0.5
        ... )
    """
    filepath = Path(filepath)
    
    # Create checkpoint dictionary with version metadata
    checkpoint = {
        # Version information
        'version': '1.0.0',
        'pytorch_version': torch.__version__,
        'python_version': sys.version,
        
        # Training state
        'epoch': epoch,
        'global_step': step,
        'best_val_loss': best_val_loss,
        'is_best': is_best,
        
        # Model state
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        
        # Model architecture details for reconstruction
        'model_architecture': {
            'input_channels': model.input_channels,
            'hidden_channels': model.hidden_channels,
            'output_channels': model.output_channels,
            'kernel_size': model.kernel_size,
            'use_attention': getattr(model, 'use_attention', False),
            'use_group_norm': getattr(model, 'use_group_norm', False),
            'dropout_rate': getattr(model, 'dropout_rate', 0.0),
            'use_spatial_dropout': getattr(model, 'use_spatial_dropout', True),
            'model_type': getattr(config, 'model_type', None) if config else None,
        }
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # Add scheduler state if provided
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add configuration if provided
    if config is not None:
        checkpoint['config'] = config
    
    if region_config is not None:
        checkpoint['region_config'] = region_config
    
    # Add additional metadata if provided
    if additional_metadata is not None:
        checkpoint['additional_metadata'] = additional_metadata
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, filepath)
    
    logger.info(f"Saved checkpoint to {filepath}")
    logger.info(f"Checkpoint version: {checkpoint['version']}")
    logger.info(f"Epoch: {epoch}, Step: {step}, Best val loss: {best_val_loss:.4f}")


def load_model_checkpoint(
    filepath: Union[str, Path],
    model: Optional[Union[ConvLSTMUNet, 'DeepConvLSTMUNet']] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load model checkpoint with backward compatibility.
    
    This is a standalone function for loading model checkpoints that can be used
    independently of the ConvLSTMTrainer class. It loads model parameters and
    optionally restores optimizer and scheduler state.
    
    Args:
        filepath: Path to checkpoint file
        model: Optional ConvLSTMUNet or DeepConvLSTMUNet model to load state into
               If None, only returns checkpoint dictionary
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on (default: auto-detect)
        strict: Whether to strictly enforce state dict loading (default: True)
    
    Returns:
        Dictionary containing checkpoint data including:
        - 'epoch': Training epoch
        - 'global_step': Training step
        - 'best_val_loss': Best validation loss
        - 'config': ConvLSTMConfig if available
        - 'region_config': RegionConfig if available
        - 'model_architecture': Model architecture details
        - Additional metadata from checkpoint
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is incompatible with provided model
    
    Example:
        >>> model = ConvLSTMUNet(input_channels=56, hidden_channels=[32, 64])
        >>> checkpoint_data = load_model_checkpoint(
        ...     filepath='checkpoints/model.pt',
        ...     model=model,
        ...     device=torch.device('cuda')
        ... )
        >>> print(f"Loaded epoch {checkpoint_data['epoch']}")
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Auto-detect device if not provided
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading checkpoint from {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Check version for backward compatibility
    checkpoint_version = checkpoint.get('version', '0.0.0')
    logger.info(f"Checkpoint version: {checkpoint_version}")
    
    # Load model state if model is provided
    if model is not None:
        # Validate model architecture compatibility
        if 'model_architecture' in checkpoint:
            arch = checkpoint['model_architecture']
            
            # Check if architecture matches
            if arch['input_channels'] != model.input_channels:
                raise RuntimeError(
                    f"Model architecture mismatch: checkpoint has "
                    f"input_channels={arch['input_channels']}, "
                    f"but provided model has input_channels={model.input_channels}"
                )
            
            if arch['hidden_channels'] != model.hidden_channels:
                raise RuntimeError(
                    f"Model architecture mismatch: checkpoint has "
                    f"hidden_channels={arch['hidden_channels']}, "
                    f"but provided model has hidden_channels={model.hidden_channels}"
                )
            
            logger.info(f"Model architecture validated: {arch}")
        else:
            logger.warning(
                "Checkpoint does not contain model_architecture metadata. "
                "Skipping architecture validation."
            )
        
        # Load model state
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            logger.info("Model state loaded successfully")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load model state: {e}")
            else:
                logger.warning(f"Model state loaded with errors (strict=False): {e}")
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Optimizer state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Scheduler state loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
    
    # Extract training progress
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('global_step', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    logger.info(
        f"Checkpoint loaded: epoch={epoch}, step={step}, "
        f"best_val_loss={best_val_loss:.4f}"
    )
    
    # Return checkpoint data
    return {
        'epoch': epoch,
        'global_step': step,
        'best_val_loss': best_val_loss,
        'config': checkpoint.get('config'),
        'region_config': checkpoint.get('region_config'),
        'model_architecture': checkpoint.get('model_architecture'),
        'version': checkpoint_version,
        'is_best': checkpoint.get('is_best', False),
        'additional_metadata': checkpoint.get('additional_metadata', {})
    }


class ConvLSTMTrainer:
    """Training pipeline for ConvLSTM model.
    
    This class manages the complete training workflow including:
    - Model initialization and optimization
    - Training and validation loops
    - Gradient clipping and accumulation
    - Mixed precision training
    - Checkpointing and early stopping
    - Progress logging
    
    Attributes:
        model: ConvLSTMUNet instance
        config: ConvLSTMConfig with hyperparameters
        region_config: RegionConfig for boundaries
        normalizer: ConvLSTMNormalizer for preprocessing
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        loss_fn: WeightedPrecipitationLoss function
        scaler: GradScaler for mixed precision training
        device: Device to run training on (cuda/cpu)
        logger: Logger for training progress
    """
    
    def __init__(self,
                 model: Union[ConvLSTMUNet, DeepConvLSTMUNet],
                 config: ConvLSTMConfig,
                 region_config: RegionConfig,
                 normalizer: ConvLSTMNormalizer,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None,
                 use_masked_loss: bool = False,
                 filled_weight: float = 0.1):
        """Initialize ConvLSTMTrainer.
        
        Args:
            model: ConvLSTMUNet or DeepConvLSTMUNet instance
            config: ConvLSTMConfig with hyperparameters
            region_config: RegionConfig for boundaries
            normalizer: ConvLSTMNormalizer for preprocessing
            device: Device to run training on (default: auto-detect)
            logger: Logger instance (default: create new logger)
            use_masked_loss: Whether to use masked loss (default: False)
            filled_weight: Weight for filled/invalid values in masked loss (default: 0.1)
        """
        self.model = model
        self.config = config
        self.region_config = region_config
        self.normalizer = normalizer
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer (AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.9,      # 改为0.7，每次衰减到原来的70%（更温和）
            patience=20,     # 改为10，需要10个epoch没有改善才衰减（更宽容）
            min_lr=1e-6,     # 添加最小学习率限制
            verbose=True
        )
        
        # Initialize loss function
        self.use_masked_loss = use_masked_loss
        self.filled_weight = filled_weight
        
        if use_masked_loss:
            # Use masked loss that handles validity masks
            self.loss_fn = CombinedMaskedLoss(
                mse_weight=1.0,
                gradient_weight=0.1,
                filled_weight=filled_weight
            ).to(self.device)
            self.logger.info(f"Using masked loss with filled_weight={filled_weight}")
        else:
            # Use standard weighted precipitation loss
            lat_coords = None
            if hasattr(region_config, 'downstream_lat_range'):
                import numpy as np
                lat_min, lat_max = region_config.downstream_lat_range
                lat_coords = None
            
            self.loss_fn = WeightedPrecipitationLoss(
                high_precip_threshold=config.high_precip_threshold,
                high_precip_weight=config.high_precip_weight,
                extreme_precip_threshold=getattr(config, 'extreme_precip_threshold', 50.0),
                extreme_precip_weight=getattr(config, 'extreme_precip_weight', 10.0),
                latitude_coords=lat_coords
            ).to(self.device)
        
        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # Setup logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.logger.info(f"Initialized ConvLSTMTrainer on device: {self.device}")
        self.logger.info(f"Model type: {type(model).__name__}")
        self.logger.info(f"Model architecture: {config.hidden_channels}")
        
        # Log attention and normalization features
        if hasattr(model, 'use_attention'):
            self.logger.info(f"Self-attention enabled: {model.use_attention}")
        if hasattr(model, 'use_group_norm'):
            self.logger.info(f"Group normalization enabled: {getattr(model, 'use_group_norm', False)}")
        
        self.logger.info(f"Mixed precision training: {config.use_amp}")
        self.logger.info(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    
    def train_step(self, batch: tuple) -> float:
        """Execute one training step.
        
        Args:
            batch: Tuple of (inputs, targets) or (inputs, targets, mask) from DataLoader
                inputs: [B, T, C, H, W] or dict with 'downstream' and 'upstream' keys
                targets: [B, H, W]
                mask: [B, H, W] (optional, for masked loss)
        
        Returns:
            Loss value for this batch
        """
        # Handle both masked and unmasked batches
        if len(batch) == 3:
            inputs, targets, mask = batch
            mask = mask.to(self.device)
        else:
            inputs, targets = batch
            mask = None
        
        # Move inputs to device (handle both tensor and dict)
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        targets = targets.to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp):
            predictions = self.model(inputs)
            
            # Compute loss (with or without mask)
            if self.use_masked_loss and mask is not None:
                # Masked loss returns dict with 'total' key
                loss_dict = self.loss_fn(predictions, targets, mask)
                loss = loss_dict['total']
            else:
                loss = self.loss_fn(predictions, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Return unscaled loss for logging
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validation_step(self, batch: tuple) -> float:
        """Execute one validation step.
        
        Args:
            batch: Tuple of (inputs, targets) or (inputs, targets, mask) from DataLoader
                inputs: [B, T, C, H, W] or dict with 'downstream' and 'upstream' keys
        
        Returns:
            Loss value for this batch
        """
        # Handle both masked and unmasked batches
        if len(batch) == 3:
            inputs, targets, mask = batch
            mask = mask.to(self.device)
        else:
            inputs, targets = batch
            mask = None
        
        # Move inputs to device (handle both tensor and dict)
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)
        
        targets = targets.to(self.device)
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            with autocast(enabled=self.config.use_amp):
                predictions = self.model(inputs)
                
                # Compute loss (with or without mask)
                if self.use_masked_loss and mask is not None:
                    loss_dict = self.loss_fn(predictions, targets, mask)
                    loss = loss_dict['total']
                else:
                    loss = self.loss_fn(predictions, targets)
        
        return loss.item()
    
    def train_epoch(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Training step
            loss = self.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Validation at specified frequency
            if val_loader is not None and self.global_step % self.config.validation_frequency == 0:
                val_loss = self.validate(val_loader)
                self.logger.info(
                    f"Step {self.global_step}: train_loss={loss:.4f}, val_loss={val_loss:.4f}"
                )
                self.model.train()  # Switch back to training mode
        
        # Handle remaining gradients if batch count is not divisible by accumulation steps
        if num_batches % self.config.gradient_accumulation_steps != 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            self.global_step += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'train_loss': avg_loss,
            'num_batches': num_batches
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Run validation.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                loss = self.validation_step(batch)
                val_loss += loss
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_val_loss
    
    def train(self,
              train_dataset: ConvLSTMDataset,
              val_dataset: Optional[ConvLSTMDataset] = None,
              checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Run full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            checkpoint_dir: Directory to save checkpoints (optional)
        
        Returns:
            Dictionary with training history and best metrics
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset is not None:
            self.logger.info(f"Validation samples: {len(val_dataset)}")
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, val_loader)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if checkpoint_dir is not None:
                        self.save_checkpoint(
                            checkpoint_dir / 'best_model.pt',
                            is_best=True
                        )
                        self.logger.info(f"Saved best model with val_loss={val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
            
            # Record metrics
            history['train_loss'].append(train_metrics['train_loss'])
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch + 1}/{self.config.num_epochs}: "
            log_msg += f"train_loss={train_metrics['train_loss']:.4f}"
            if val_loader is not None:
                log_msg += f", val_loss={val_loss:.4f}"
            log_msg += f", lr={self.optimizer.param_groups[0]['lr']:.6f}"
            log_msg += f", time={epoch_time:.2f}s"
            self.logger.info(log_msg)
            
            # Periodic checkpointing
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pt'
                )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'history': history,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save training checkpoint with version metadata.
        
        Saves a comprehensive checkpoint including:
        - Model parameters and architecture configuration
        - Optimizer state and learning rate scheduler state
        - Training progress (epoch, step, best validation loss)
        - Region configuration for spatial boundaries
        - Version metadata for backward compatibility
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        import sys
        
        # Create checkpoint dictionary with version metadata
        checkpoint = {
            # Version information
            'version': '1.0.0',
            'pytorch_version': torch.__version__,
            'python_version': sys.version,
            
            # Training state
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'is_best': is_best,
            
            # Model state
            'model_state_dict': self.model.state_dict(),
            'model_type': type(self.model).__name__,
            
            # Optimizer and scheduler state
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            
            # Configuration
            'config': self.config,
            'region_config': self.region_config,
            
            # Model architecture details for reconstruction
            'model_architecture': {
                'input_channels': self.model.input_channels,
                'hidden_channels': self.model.hidden_channels,
                'output_channels': self.model.output_channels,
                'kernel_size': self.model.kernel_size,
                'use_attention': getattr(self.model, 'use_attention', False),
                'use_group_norm': getattr(self.model, 'use_group_norm', False),
            }
        }
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint, path)
        
        self.logger.info(f"Saved checkpoint to {path}")
        self.logger.info(f"Checkpoint version: {checkpoint['version']}")
        self.logger.info(f"Epoch: {self.current_epoch}, Step: {self.global_step}, "
                        f"Best val loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, path: Path, strict: bool = True):
        """Load training checkpoint with backward compatibility.
        
        Loads a checkpoint and restores all training state. Supports backward
        compatibility by checking version metadata and handling missing keys
        gracefully.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict loading (default: True)
                   Set to False for backward compatibility with older checkpoints
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is incompatible with current model
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        self.logger.info(f"Loading checkpoint from {path}")
        
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check version for backward compatibility
        checkpoint_version = checkpoint.get('version', '0.0.0')
        self.logger.info(f"Checkpoint version: {checkpoint_version}")
        
        # Validate model architecture compatibility
        if 'model_architecture' in checkpoint:
            arch = checkpoint['model_architecture']
            
            # Check if architecture matches
            if arch['input_channels'] != self.model.input_channels:
                raise RuntimeError(
                    f"Model architecture mismatch: checkpoint has "
                    f"input_channels={arch['input_channels']}, "
                    f"but current model has input_channels={self.model.input_channels}"
                )
            
            if arch['hidden_channels'] != self.model.hidden_channels:
                raise RuntimeError(
                    f"Model architecture mismatch: checkpoint has "
                    f"hidden_channels={arch['hidden_channels']}, "
                    f"but current model has hidden_channels={self.model.hidden_channels}"
                )
            
            self.logger.info(f"Model architecture validated: {arch}")
        else:
            self.logger.warning(
                "Checkpoint does not contain model_architecture metadata. "
                "Skipping architecture validation."
            )
        
        # Load model state
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            self.logger.info("Model state loaded successfully")
        except Exception as e:
            if strict:
                raise RuntimeError(f"Failed to load model state: {e}")
            else:
                self.logger.warning(f"Model state loaded with errors (strict=False): {e}")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("Optimizer state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")
        else:
            self.logger.warning("Checkpoint does not contain optimizer state")
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")
        else:
            self.logger.warning("Checkpoint does not contain scheduler state")
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                self.logger.info("GradScaler state loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load scaler state: {e}")
        else:
            self.logger.warning("Checkpoint does not contain scaler state")
        
        # Restore training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.epochs_without_improvement = 0  # Reset early stopping counter
        
        self.logger.info(
            f"Training state restored: epoch={self.current_epoch}, "
            f"step={self.global_step}, best_val_loss={self.best_val_loss:.4f}"
        )
        
        # Log checkpoint configuration for reference
        if 'config' in checkpoint:
            self.logger.info(f"Checkpoint config: {checkpoint['config']}")
        if 'region_config' in checkpoint:
            self.logger.info(f"Checkpoint region config: {checkpoint['region_config']}")
    
    def reset_learning_rate(self, new_lr: float):
        """Reset learning rate to a new value.
        
        Useful when resuming training from a checkpoint with a different
        learning rate than what was saved.
        
        Args:
            new_lr: New learning rate to set
        """
        old_lr = self.optimizer.param_groups[0]['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.logger.info(f"Learning rate reset: {old_lr:.6f} -> {new_lr:.6f}")
        
        self.logger.info("Checkpoint loaded successfully")
