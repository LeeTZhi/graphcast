"""Configuration dataclasses for ConvLSTM weather prediction."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ConvLSTMConfig:
    """Configuration for ConvLSTM model and training.
    
    This configuration class contains all hyperparameters for the ConvLSTM U-Net
    architecture and training pipeline, designed to work within 12GB GPU memory
    constraints.
    
    Attributes:
        # Model architecture
        input_channels: Number of input channels (5 vars × 11 levels + 1 precip = 56)
        hidden_channels: List of hidden dimensions for encoder and bottleneck layers
        kernel_size: Size of convolutional kernel for ConvLSTM cells
        output_channels: Number of output channels (1 for precipitation)
        
        # Training hyperparameters
        learning_rate: Initial learning rate for AdamW optimizer
        batch_size: Number of samples per batch
        num_epochs: Maximum number of training epochs
        gradient_clip_norm: Maximum gradient norm for clipping
        weight_decay: L2 regularization coefficient
        
        # Data configuration
        window_size: Number of historical timesteps used as input (default: 6 = 3 days)
        target_offset: Number of timesteps ahead to predict (default: 1 = 12 hours)
        
        # Loss function parameters
        high_precip_threshold: Precipitation threshold (mm) for high-weight events
        high_precip_weight: Weight multiplier for high precipitation events
        
        # Memory optimization
        use_amp: Whether to use automatic mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_workers: Number of DataLoader worker processes for data prefetching
        
        # Checkpointing and validation
        checkpoint_frequency: Save checkpoint every N training steps
        validation_frequency: Run validation every N training steps
        early_stopping_patience: Stop training after N epochs without improvement
    """
    
    # Model architecture
    input_channels: int = 56  # 5 vars × 11 levels + 1 precip
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 3
    output_channels: int = 1
    
    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 4
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    
    # Data configuration
    window_size: int = 6  # 3 days at 12-hour intervals
    target_offset: int = 1  # Predict 12 hours ahead
    
    # Loss function parameters
    high_precip_threshold: float = 10.0  # mm
    high_precip_weight: float = 3.0
    
    # Memory optimization
    use_amp: bool = True  # Automatic mixed precision
    gradient_accumulation_steps: int = 1
    num_workers: int = 2
    
    # Checkpointing and validation
    checkpoint_frequency: int = 1000
    validation_frequency: int = 500
    early_stopping_patience: int = 10
    
    def validate(self) -> None:
        """Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Validate model architecture
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {self.input_channels}")
        
        if not self.hidden_channels or len(self.hidden_channels) < 1:
            raise ValueError("hidden_channels must contain at least one value")
        
        if any(h <= 0 for h in self.hidden_channels):
            raise ValueError(f"All hidden_channels must be positive, got {self.hidden_channels}")
        
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive and odd, got {self.kernel_size}")
        
        if self.output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {self.output_channels}")
        
        # Validate training hyperparameters
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_clip_norm <= 0:
            raise ValueError(f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}")
        
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        
        # Validate data configuration
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")
        
        if self.target_offset <= 0:
            raise ValueError(f"target_offset must be positive, got {self.target_offset}")
        
        # Validate loss function parameters
        if self.high_precip_threshold < 0:
            raise ValueError(f"high_precip_threshold must be non-negative, got {self.high_precip_threshold}")
        
        if self.high_precip_weight < 1.0:
            raise ValueError(f"high_precip_weight must be >= 1.0, got {self.high_precip_weight}")
        
        # Validate memory optimization
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"gradient_accumulation_steps must be positive, got {self.gradient_accumulation_steps}")
        
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
        
        # Validate checkpointing and validation
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}")
        
        if self.validation_frequency <= 0:
            raise ValueError(f"validation_frequency must be positive, got {self.validation_frequency}")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
