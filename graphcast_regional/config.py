# Copyright 2024 Regional Weather Prediction Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration dataclasses for Regional Weather Prediction System."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RegionConfig:
    """Configuration for regional boundaries and graph connectivity.
    
    Defines the upstream and downstream regions for weather prediction,
    along with parameters for graph construction.
    
    Attributes:
        downstream_lat_min: Minimum latitude for downstream region (degrees).
        downstream_lat_max: Maximum latitude for downstream region (degrees).
        downstream_lon_min: Minimum longitude for downstream region (degrees).
        downstream_lon_max: Maximum longitude for downstream region (degrees).
        upstream_lat_min: Minimum latitude for upstream region (degrees).
        upstream_lat_max: Maximum latitude for upstream region (degrees).
        upstream_lon_min: Minimum longitude for upstream region (degrees).
        upstream_lon_max: Maximum longitude for upstream region (degrees).
        intra_domain_k_neighbors: Number of nearest neighbors for intra-domain edges.
        inter_domain_k_neighbors: Number of nearest neighbors for inter-domain edges.
        boundary_threshold_lon: Longitude threshold for identifying boundary nodes.
    """
    # Downstream region boundaries (target prediction area)
    downstream_lat_min: float = 25.0
    downstream_lat_max: float = 40.0
    downstream_lon_min: float = 110.0
    downstream_lon_max: float = 125.0
    
    # Upstream region boundaries (influencing area)
    upstream_lat_min: float = 25.0
    upstream_lat_max: float = 50.0
    upstream_lon_min: float = 70.0
    upstream_lon_max: float = 110.0
    
    # Graph connectivity parameters
    intra_domain_k_neighbors: int = 8
    inter_domain_k_neighbors: int = 32
    boundary_threshold_lon: float = 111.0
    
    def __post_init__(self):
        """Validate region configuration."""
        # Validate downstream region
        if self.downstream_lat_min >= self.downstream_lat_max:
            raise ValueError(
                f"downstream_lat_min ({self.downstream_lat_min}) must be less than "
                f"downstream_lat_max ({self.downstream_lat_max})"
            )
        if self.downstream_lon_min >= self.downstream_lon_max:
            raise ValueError(
                f"downstream_lon_min ({self.downstream_lon_min}) must be less than "
                f"downstream_lon_max ({self.downstream_lon_max})"
            )
        
        # Validate upstream region
        if self.upstream_lat_min >= self.upstream_lat_max:
            raise ValueError(
                f"upstream_lat_min ({self.upstream_lat_min}) must be less than "
                f"upstream_lat_max ({self.upstream_lat_max})"
            )
        if self.upstream_lon_min >= self.upstream_lon_max:
            raise ValueError(
                f"upstream_lon_min ({self.upstream_lon_min}) must be less than "
                f"upstream_lon_max ({self.upstream_lon_max})"
            )
        
        # Validate k-neighbors
        if self.intra_domain_k_neighbors <= 0:
            raise ValueError(
                f"intra_domain_k_neighbors must be positive, got {self.intra_domain_k_neighbors}"
            )
        if self.inter_domain_k_neighbors <= 0:
            raise ValueError(
                f"inter_domain_k_neighbors must be positive, got {self.inter_domain_k_neighbors}"
            )


@dataclass
class ModelConfig:
    """Configuration for Regional GNN model architecture.
    
    Defines the hyperparameters for the encoder-processor-decoder architecture.
    
    Attributes:
        latent_size: Dimension of latent node representations.
        num_gnn_layers: Number of message passing layers in the processor.
        mlp_hidden_size: Hidden layer size for MLPs.
        mlp_num_hidden_layers: Number of hidden layers in MLPs.
        use_residual: Whether to use residual connections in GNN layers.
        activation: Activation function name ('swish', 'relu', 'gelu').
        use_layer_norm: Whether to use layer normalization.
        dropout_rate: Dropout rate for training (0.0 means no dropout).
        edge_feature_size: Size of edge features (computed from spatial features).
    """
    # Model architecture
    latent_size: int = 256
    num_gnn_layers: int = 12
    mlp_hidden_size: int = 256
    mlp_num_hidden_layers: int = 2
    
    # Training and regularization
    use_residual: bool = True
    activation: str = "swish"
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    
    # Feature dimensions
    edge_feature_size: Optional[int] = None  # Computed from graph builder
    
    def __post_init__(self):
        """Validate model configuration."""
        if self.latent_size <= 0:
            raise ValueError(f"latent_size must be positive, got {self.latent_size}")
        
        if self.num_gnn_layers <= 0:
            raise ValueError(f"num_gnn_layers must be positive, got {self.num_gnn_layers}")
        
        if self.mlp_hidden_size <= 0:
            raise ValueError(f"mlp_hidden_size must be positive, got {self.mlp_hidden_size}")
        
        if self.mlp_num_hidden_layers < 0:
            raise ValueError(
                f"mlp_num_hidden_layers must be non-negative, got {self.mlp_num_hidden_layers}"
            )
        
        if self.activation not in ["swish", "relu", "gelu", "elu", "tanh"]:
            raise ValueError(
                f"activation must be one of ['swish', 'relu', 'gelu', 'elu', 'tanh'], "
                f"got {self.activation}"
            )
        
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}"
            )


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        learning_rate: Initial learning rate for optimizer.
        batch_size: Number of samples per training batch.
        num_epochs: Number of training epochs.
        gradient_clip_norm: Maximum gradient norm for clipping.
        weight_decay: Weight decay coefficient for AdamW.
        warmup_steps: Number of warmup steps for learning rate schedule.
        high_precip_threshold: Threshold (mm) for high precipitation weighting.
        high_precip_weight: Weight multiplier for high precipitation samples.
        validation_frequency: Validate every N training steps.
        checkpoint_frequency: Save checkpoint every N training steps.
        early_stopping_patience: Stop if no improvement for N validations.
    """
    # Optimizer settings
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Loss function settings
    high_precip_threshold: float = 10.0
    high_precip_weight: float = 3.0
    
    # Training loop settings
    validation_frequency: int = 500
    checkpoint_frequency: int = 1000
    early_stopping_patience: int = 100
    
    def __post_init__(self):
        """Validate training configuration."""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        
        if self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive, got {self.gradient_clip_norm}"
            )
        
        if self.high_precip_threshold < 0:
            raise ValueError(
                f"high_precip_threshold must be non-negative, got {self.high_precip_threshold}"
            )
        
        if self.high_precip_weight <= 0:
            raise ValueError(
                f"high_precip_weight must be positive, got {self.high_precip_weight}"
            )
