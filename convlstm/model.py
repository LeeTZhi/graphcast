"""ConvLSTM U-Net model for precipitation prediction.

This module implements the core ConvLSTM architecture including:
- ConvLSTMCell: Single cell for processing one timestep
- SelfAttention: Self-attention module for capturing long-range dependencies
- ConvLSTMUNet: Full encoder-decoder architecture with skip connections
- WeightedPrecipitationLoss: Custom loss function with precipitation and latitude weighting
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional


class SelfAttention(nn.Module):
    """Self-Attention Block (Non-local Neural Network style).
    
    This module captures long-range dependencies by calculating the relationship
    between every pixel and every other pixel in the feature map.
    
    Attributes:
        in_channels: Number of input channels
        inter_channels: Reduced channels for Q and K (typically in_channels // 8)
        gamma: Learnable scale parameter for attention output
    """
    
    def __init__(self, in_channels: int):
        """Initialize SelfAttention.
        
        Args:
            in_channels: Number of input channels
        """
        super(SelfAttention, self).__init__()
        
        self.in_channels = in_channels
        # Q, K usually reduce channels to save memory (e.g., 1/8)
        self.inter_channels = in_channels // 8 if in_channels >= 8 else in_channels // 2

        # 1x1 Convs to generate Query, Key, Value
        self.query_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scale parameter, initialized to 0
        # This allows the network to start with local features and gradually learn global attention
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.
        
        Args:
            x: Input feature map [Batch, Channel, Height, Width]
            
        Returns:
            out: Attention-enhanced feature map [B, C, H, W]
        """
        batch_size, C, H, W = x.size()
        N = H * W  # Total number of spatial locations

        # --- Query ---
        # proj_query: [B, C', H, W] -> view [B, C', N] -> permute [B, N, C']
        proj_query = self.query_conv(x).view(batch_size, -1, N).permute(0, 2, 1)

        # --- Key ---
        # proj_key: [B, C', H, W] -> view [B, C', N]
        proj_key = self.key_conv(x).view(batch_size, -1, N)

        # --- Attention Map ---
        # Energy: [B, N, N] (Relationship between every pixel i and j)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # Normalize to probability

        # --- Value ---
        # proj_value: [B, C, H, W] -> view [B, C, N]
        proj_value = self.value_conv(x).view(batch_size, -1, N)

        # --- Output ---
        # out: [B, C, N] -> view [B, C, H, W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell for spatio-temporal processing.
    
    This cell processes one timestep of spatial data, applying convolutions
    in both input-to-state and state-to-state transitions. It implements
    the standard LSTM gates (input, forget, output, cell) using spatial
    convolutions to preserve spatial structure.
    
    Attributes:
        input_dim: Number of input channels
        hidden_dim: Number of hidden state channels
        kernel_size: Size of convolutional kernel
        bias: Whether to use bias in convolutions
        padding: Padding size to maintain spatial dimensions
        use_group_norm: Whether to use Group Normalization
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, bias: bool = True,
                 use_group_norm: bool = True):
        """Initialize ConvLSTMCell.
        
        Args:
            input_dim: Number of input channels
            hidden_dim: Number of hidden state channels
            kernel_size: Size of convolutional kernel (default: 3)
            bias: Whether to use bias in convolutions (default: True)
            use_group_norm: Whether to use Group Normalization (default: True)
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.use_group_norm = use_group_norm
        
        # Single convolution that computes all 4 gates at once
        # Input: concatenation of input and hidden state
        # Output: 4 * hidden_dim channels (for i, f, o, g gates)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias
        )
        
        # Group Normalization (recommended over Batch Normalization for small batches)
        if use_group_norm:
            # num_groups must divide 4*hidden_dim
            num_groups = min(16, 4 * hidden_dim)
            while (4 * hidden_dim) % num_groups != 0:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=4 * hidden_dim)
    
    def forward(self, input_tensor: torch.Tensor, 
                cur_state: Tuple[torch.Tensor, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.
        
        Args:
            input_tensor: Input at current timestep [B, C_in, H, W]
            cur_state: Tuple of (h_cur, c_cur) hidden states
                h_cur: Hidden state [B, hidden_dim, H, W]
                c_cur: Cell state [B, hidden_dim, H, W]
            
        Returns:
            Tuple of (h_next, c_next) updated hidden states
                h_next: Updated hidden state [B, hidden_dim, H, W]
                c_next: Updated cell state [B, hidden_dim, H, W]
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution to compute all gates
        combined_conv = self.conv(combined)
        
        # Apply Group Normalization if enabled
        if self.use_group_norm:
            combined_conv = self.norm(combined_conv)
        
        # Split into 4 gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, height: int, width: int, 
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros.
        
        Args:
            batch_size: Batch size
            height: Spatial height
            width: Spatial width
            device: Device to create tensors on
            
        Returns:
            Tuple of (h_0, c_0) initialized hidden states
        """
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTMUNet(nn.Module):
    """ConvLSTM U-Net for precipitation prediction.
    
    This model implements an encoder-decoder architecture with ConvLSTM layers
    to capture spatio-temporal patterns in atmospheric data. The U-Net structure
    with skip connections ensures the output maintains the same spatial resolution
    as the input, which is essential for dense precipitation prediction.
    
    Architecture:
        - Encoder: ConvLSTM layer + downsampling (MaxPool2d)
        - Bottleneck: ConvLSTM layer at reduced resolution
        - Decoder: Upsampling + skip connections + ConvLSTM layer
        - Output head: Conv2d 1x1 with ReLU for non-negative precipitation
    
    Attributes:
        input_channels: Number of input channels (56 per timestep)
        hidden_channels: List of hidden dimensions [encoder, bottleneck]
        output_channels: Number of output channels (1 for precipitation)
        kernel_size: Size of convolutional kernel for ConvLSTM cells
    """
    
    def __init__(self, input_channels: int = 56, 
                 hidden_channels: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: int = 3,
                 use_attention: bool = True,
                 use_group_norm: bool = True,
                 multi_variable: bool = False):
        """Initialize ConvLSTMUNet.
        
        Args:
            input_channels: Number of input channels (default: 56)
            hidden_channels: List of hidden dimensions (default: [32, 64])
            output_channels: Number of output channels (default: 1, ignored if multi_variable=True)
            kernel_size: Size of convolutional kernel (default: 3)
            use_attention: Whether to use self-attention at bottleneck (default: True)
            use_group_norm: Whether to use Group Normalization in ConvLSTM cells (default: True)
            multi_variable: Whether to predict all variables (56 channels) or just precipitation (1 channel) (default: False)
        """
        super(ConvLSTMUNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64]
        
        if len(hidden_channels) < 2:
            raise ValueError("hidden_channels must contain at least 2 values for encoder and bottleneck")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.multi_variable = multi_variable
        self.output_channels = 56 if multi_variable else output_channels
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.use_group_norm = use_group_norm
        
        # Encoder: ConvLSTM layer
        self.encoder = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Downsampling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck: ConvLSTM layer at reduced resolution
        self.bottleneck = ConvLSTMCell(
            input_dim=hidden_channels[0],
            hidden_dim=hidden_channels[1],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Self-Attention at bottleneck (applied after bottleneck processing)
        if use_attention:
            self.attention = SelfAttention(in_channels=hidden_channels[1])
        
        # Upsampling layer (not used directly - see forward() for F.interpolate usage)
        # We use F.interpolate in forward() to handle odd spatial dimensions correctly
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Decoder: ConvLSTM layer with skip connection
        # Input is bottleneck output + encoder output (skip connection)
        self.decoder = ConvLSTMCell(
            input_dim=hidden_channels[1] + hidden_channels[0],
            hidden_dim=hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Output head: 1x1 convolution to map to precipitation (or all variables in multi-variable mode)
        self.output_head = nn.Conv2d(
            in_channels=hidden_channels[0],
            out_channels=self.output_channels,
            kernel_size=1
        )
        
        # ReLU activation for non-negative precipitation
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor [B, T, C, H, W] or dict with 'downstream' key
               B = batch size
               T = time steps (default 6)
               C = channels (56)
               H, W = spatial dimensions (can be odd or even)
               
        Returns:
            Predicted precipitation [B, 1, H, W]
            
        Note:
            This implementation handles odd spatial dimensions correctly by using
            F.interpolate with exact target size instead of fixed scale_factor.
            This prevents size mismatches when concatenating skip connections.
            
            If x is a dict (dual-stream mode), only uses 'downstream' key.
        """
        # Handle dict input (dual-stream mode) - use only downstream
        if isinstance(x, dict):
            x = x['downstream']
        
        batch_size, time_steps, channels, height, width = x.size()
        device = x.device
        
        # Initialize hidden states for encoder
        h_enc, c_enc = self.encoder.init_hidden(batch_size, height, width, device)
        
        # Initialize hidden states for bottleneck (at reduced resolution)
        h_bot, c_bot = self.bottleneck.init_hidden(
            batch_size, height // 2, width // 2, device
        )
        
        # Process sequence through encoder and bottleneck
        for t in range(time_steps):
            # Get current timestep input
            current_input = x[:, t, :, :, :]
            
            # Encoder: Process through ConvLSTM
            h_enc, c_enc = self.encoder(current_input, (h_enc, c_enc))
            
            # Downsample encoder output
            encoded_feat = self.pool(h_enc)
            
            # Bottleneck: Process through ConvLSTM at reduced resolution
            h_bot, c_bot = self.bottleneck(encoded_feat, (h_bot, c_bot))
            
            # Apply Self-Attention at bottleneck (captures global context at each timestep)
            if self.use_attention:
                h_bot = self.attention(h_bot)
        
        # Decoder: Upsample and apply skip connection
        # Upsample bottleneck output to match encoder output size
        # Use F.interpolate with exact size to handle odd dimensions
        upsampled = torch.nn.functional.interpolate(
            h_bot,
            size=(h_enc.size(2), h_enc.size(3)),  # Match encoder spatial dimensions
            mode='bilinear',
            align_corners=True
        )
        
        # Skip connection: Concatenate with encoder output
        concat_feat = torch.cat([upsampled, h_enc], dim=1)
        
        # Initialize decoder hidden states
        h_dec, c_dec = self.decoder.init_hidden(batch_size, height, width, device)
        
        # Process through decoder ConvLSTM
        h_dec, c_dec = self.decoder(concat_feat, (h_dec, c_dec))
        
        # Output head: Map to precipitation
        output = self.output_head(h_dec)
        
        # NOTE: Do NOT apply ReLU here when working with normalized data
        # The target values are normalized (log1p + Z-score), which can be negative
        # ReLU will be applied during denormalization if needed
        # output = self.relu(output)
        
        return output



class MultiVariableLoss(nn.Module):
    """Loss function for multi-variable prediction.
    
    Combines:
    1. Weighted precipitation loss (existing WeightedPrecipitationLoss logic)
    2. MSE loss for atmospheric variables
    
    Total loss = precip_weight * precip_loss + atmos_loss
    
    Attributes:
        precip_loss_weight: Weight multiplier for precipitation vs atmospheric variables
        precip_loss_fn: WeightedPrecipitationLoss instance for precipitation channel
        atmos_loss_fn: MSE loss for atmospheric variables
    """
    
    def __init__(self, 
                 precip_loss_weight: float = 10.0,
                 high_precip_threshold: float = 10.0,
                 high_precip_weight: float = 5.0,
                 extreme_precip_threshold: float = 50.0,
                 extreme_precip_weight: float = 10.0,
                 latitude_coords: Optional[np.ndarray] = None):
        """Initialize MultiVariableLoss.
        
        Args:
            precip_loss_weight: Weight for precipitation vs atmospheric variables (default: 10.0)
            high_precip_threshold: Threshold (mm) for high precipitation (default: 10.0)
            high_precip_weight: Weight multiplier for high precipitation (default: 5.0)
            extreme_precip_threshold: Threshold (mm) for extreme precipitation (default: 50.0)
            extreme_precip_weight: Weight multiplier for extreme precipitation (default: 10.0)
            latitude_coords: Array of latitude values in degrees (default: None)
        """
        super().__init__()
        
        self.precip_loss_weight = precip_loss_weight
        
        # Reuse existing WeightedPrecipitationLoss for precipitation channel
        self.precip_loss_fn = WeightedPrecipitationLoss(
            high_precip_threshold=high_precip_threshold,
            high_precip_weight=high_precip_weight,
            extreme_precip_threshold=extreme_precip_threshold,
            extreme_precip_weight=extreme_precip_weight,
            latitude_coords=latitude_coords
        )
        
        # Simple MSE for atmospheric variables
        self.atmos_loss_fn = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute multi-variable loss.
        
        Args:
            predictions: Predicted values [B, C, H, W] where C is 1 or 56
            targets: Ground truth values [B, H, W] (single-variable) or [B, C, H, W] (multi-variable)
        
        Returns:
            Weighted loss scalar
        """
        # Handle single-variable mode (backward compatibility)
        if predictions.size(1) == 1:
            return self.precip_loss_fn(predictions, targets)
        
        # Multi-variable mode
        # Channel 55 is precipitation, channels 0-54 are atmospheric variables
        precip_pred = predictions[:, 55:56, :, :]  # [B, 1, H, W]
        
        # Handle targets shape - could be [B, 56, H, W] or [B, H, W]
        if targets.dim() == 3:
            # Single-channel target (backward compatibility)
            precip_target = targets  # [B, H, W]
            # For atmospheric variables, we can't compute loss without targets
            # This shouldn't happen in multi-variable mode, but handle gracefully
            return self.precip_loss_fn(precip_pred, precip_target)
        else:
            # Multi-channel target [B, 56, H, W]
            precip_target = targets[:, 55, :, :]  # [B, H, W]
            atmos_pred = predictions[:, :55, :, :]  # [B, 55, H, W]
            atmos_target = targets[:, :55, :, :]  # [B, 55, H, W]
        
        # Compute losses
        precip_loss = self.precip_loss_fn(precip_pred, precip_target)
        atmos_loss = self.atmos_loss_fn(atmos_pred, atmos_target)
        
        # Weighted combination
        total_loss = self.precip_loss_weight * precip_loss + atmos_loss
        
        return total_loss


class WeightedPrecipitationLoss(nn.Module):
    """Weighted MSE loss for precipitation prediction.
    
    This loss function applies two types of weighting to focus the model on
    important predictions:
    1. Precipitation-based weighting: Higher weights for grid points with
       precipitation above thresholds (to focus on high precipitation events)
       - Moderate precipitation (>10mm): medium weight
       - Heavy precipitation (>50mm): high weight
    2. Latitude-based weighting: Weights proportional to cos(latitude) to
       account for grid cell area variation (cells are smaller near poles)
    
    The weights are normalized to have unit mean to maintain numerical stability
    and ensure the loss magnitude is comparable to standard MSE.
    
    Attributes:
        high_precip_threshold: Precipitation threshold (mm) for high-weight events
        high_precip_weight: Weight multiplier for high precipitation events
        extreme_precip_threshold: Precipitation threshold (mm) for extreme events
        extreme_precip_weight: Weight multiplier for extreme precipitation events
        latitude_coords: Latitude values for computing area-based weights
        latitude_weights: Pre-computed latitude weights (cos(latitude))
    """
    
    def __init__(self, 
                 high_precip_threshold: float = 10.0,
                 high_precip_weight: float = 5.0,
                 extreme_precip_threshold: float = 50.0,
                 extreme_precip_weight: float = 10.0,
                 latitude_coords: Optional[np.ndarray] = None):
        """Initialize WeightedPrecipitationLoss.
        
        Args:
            high_precip_threshold: Threshold (mm) for high precipitation (default: 10.0)
            high_precip_weight: Weight multiplier for high precipitation (default: 5.0, increased from 3.0)
            extreme_precip_threshold: Threshold (mm) for extreme precipitation (default: 50.0)
            extreme_precip_weight: Weight multiplier for extreme precipitation (default: 10.0)
            latitude_coords: Array of latitude values in degrees (default: None)
                If None, latitude weighting is disabled
        """
        super(WeightedPrecipitationLoss, self).__init__()
        
        if high_precip_threshold < 0:
            raise ValueError(f"high_precip_threshold must be non-negative, got {high_precip_threshold}")
        
        if high_precip_weight < 1.0:
            raise ValueError(f"high_precip_weight must be >= 1.0, got {high_precip_weight}")
        
        if extreme_precip_threshold < high_precip_threshold:
            raise ValueError(
                f"extreme_precip_threshold ({extreme_precip_threshold}) must be >= "
                f"high_precip_threshold ({high_precip_threshold})"
            )
        
        if extreme_precip_weight < high_precip_weight:
            raise ValueError(
                f"extreme_precip_weight ({extreme_precip_weight}) must be >= "
                f"high_precip_weight ({high_precip_weight})"
            )
        
        self.high_precip_threshold = high_precip_threshold
        self.high_precip_weight = high_precip_weight
        self.extreme_precip_threshold = extreme_precip_threshold
        self.extreme_precip_weight = extreme_precip_weight
        
        # Pre-compute latitude weights if coordinates provided
        if latitude_coords is not None:
            # Convert latitude to radians and compute cos(latitude)
            lat_rad = np.deg2rad(latitude_coords)
            lat_weights = np.cos(lat_rad)
            
            # Ensure weights are positive (cos can be negative near poles)
            lat_weights = np.abs(lat_weights)
            
            # Convert to tensor and store as buffer (not a parameter)
            self.register_buffer(
                'latitude_weights',
                torch.from_numpy(lat_weights).float()
            )
        else:
            self.latitude_weights = None
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE loss.
        
        Args:
            predictions: Predicted precipitation [B, 1, H, W] or [B, H, W]
            targets: Ground truth precipitation [B, H, W]
            
        Returns:
            Scalar loss value (weighted mean squared error)
        """
        # Handle predictions with channel dimension [B, 1, H, W]
        if predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)  # [B, H, W]
        
        # Ensure predictions and targets have the same shape
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            )
        
        batch_size, height, width = targets.shape
        device = targets.device
        
        # Compute squared errors
        squared_errors = (predictions - targets) ** 2
        
        # Initialize weights as ones
        weights = torch.ones_like(targets)
        
        # Apply precipitation-based weighting with multiple thresholds
        # High precipitation events (10-50mm) get higher weights
        high_precip_mask = (targets > self.high_precip_threshold) & (targets <= self.extreme_precip_threshold)
        weights[high_precip_mask] = self.high_precip_weight
        
        # Extreme precipitation events (>50mm) get even higher weights
        extreme_precip_mask = targets > self.extreme_precip_threshold
        weights[extreme_precip_mask] = self.extreme_precip_weight
        
        # Apply latitude-based weighting if available
        if self.latitude_weights is not None:
            # Latitude weights have shape [H]
            # Expand to [B, H, W] to match targets
            lat_weights_expanded = self.latitude_weights.view(1, -1, 1).expand(
                batch_size, height, width
            ).to(device)
            
            # Multiply with existing weights
            weights = weights * lat_weights_expanded
        
        # Normalize weights to unit mean
        # This ensures the loss magnitude is comparable to standard MSE
        weight_mean = weights.mean()
        if weight_mean > 0:
            weights = weights / weight_mean
        
        # Compute weighted MSE
        weighted_squared_errors = weights * squared_errors
        loss = weighted_squared_errors.mean()
        
        return loss
