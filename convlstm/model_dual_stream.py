"""Dual-stream ConvLSTM U-Net for multi-region weather prediction.

This module implements a dual-stream architecture that processes upstream and
downstream regions separately through independent encoder streams, then fuses
them at the bottleneck level for joint prediction.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Union
from convlstm.model import ConvLSTMCell, SelfAttention


class DualStreamConvLSTMUNet(nn.Module):
    """Dual-stream ConvLSTM U-Net for multi-region prediction.
    
    This architecture processes two regions (upstream and downstream) with
    different spatial dimensions through separate encoder streams. The encoded
    features are fused at the bottleneck level, allowing the model to learn
    relationships between regions without requiring spatial alignment.
    
    Architecture:
        - Downstream encoder: Processes target region
        - Upstream encoder: Processes influencing region (optional)
        - Bottleneck fusion: Concatenates encoded features from both streams
        - Decoder: Reconstructs downstream region prediction
        - Output: Precipitation prediction for downstream region only
    
    Key advantages:
        - Handles regions with different spatial dimensions
        - Learns region-specific features independently
        - Fuses information at abstract feature level
        - Maintains spatial resolution for downstream prediction
    """
    
    def __init__(self, 
                 input_channels: int = 56,
                 hidden_channels: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: int = 3,
                 use_attention: bool = True,
                 use_group_norm: bool = True,
                 dropout_rate: float = 0.0):
        """Initialize DualStreamConvLSTMUNet.
        
        Args:
            input_channels: Number of input channels (default: 56)
            hidden_channels: List of hidden dims [enc, bottleneck] (default: [64, 128])
            output_channels: Number of output channels (default: 1)
            kernel_size: Convolutional kernel size (default: 3)
            use_attention: Whether to use self-attention at bottleneck (default: True)
            use_group_norm: Whether to use group normalization (default: True)
            dropout_rate: Dropout probability (default: 0.0)
        """
        super(DualStreamConvLSTMUNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128]
        
        if len(hidden_channels) < 2:
            raise ValueError("hidden_channels must contain at least 2 values")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # Downstream encoder (always present)
        self.downstream_encoder = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Upstream encoder (for dual-stream mode)
        self.upstream_encoder = ConvLSTMCell(
            input_dim=input_channels,
            hidden_dim=hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck (input can be single or dual stream)
        # Single stream: hidden_channels[0]
        # Dual stream: 2 * hidden_channels[0] (concatenated)
        self.bottleneck_single = ConvLSTMCell(
            input_dim=hidden_channels[0],
            hidden_dim=hidden_channels[1],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        self.bottleneck_dual = ConvLSTMCell(
            input_dim=2 * hidden_channels[0],  # Concatenated features
            hidden_dim=hidden_channels[1],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Self-attention at bottleneck
        if use_attention:
            self.attention = SelfAttention(in_channels=hidden_channels[1])
        
        # Dropout at bottleneck
        if dropout_rate > 0:
            self.bottleneck_dropout = nn.Dropout2d(dropout_rate)
        
        # Decoder (reconstructs downstream region)
        self.decoder = ConvLSTMCell(
            input_dim=hidden_channels[1] + hidden_channels[0],  # Upsampled + skip
            hidden_dim=hidden_channels[0],
            kernel_size=kernel_size,
            bias=True,
            use_group_norm=use_group_norm
        )
        
        # Output head
        self.output_head = nn.Conv2d(
            in_channels=hidden_channels[0],
            out_channels=output_channels,
            kernel_size=1
        )
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass through dual-stream network.
        
        Args:
            x: Input data, can be:
               - Single tensor [B, T, C, H, W] for single-stream mode
               - Dict with keys 'downstream' and 'upstream' for dual-stream mode:
                 - 'downstream': [B, T, C, H_down, W_down]
                 - 'upstream': [B, T, C, H_up, W_up]
        
        Returns:
            Predicted precipitation [B, 1, H_down, W_down]
        """
        # Determine if dual-stream or single-stream
        is_dual_stream = isinstance(x, dict)
        
        if is_dual_stream:
            return self._forward_dual_stream(x)
        else:
            return self._forward_single_stream(x)
    
    def _forward_single_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single-stream (downstream only).
        
        Args:
            x: Input tensor [B, T, C, H, W]
        
        Returns:
            Predicted precipitation [B, 1, H, W]
        """
        batch_size, time_steps, channels, height, width = x.size()
        device = x.device
        
        # Initialize downstream encoder
        h_down, c_down = self.downstream_encoder.init_hidden(
            batch_size, height, width, device
        )
        
        # Initialize bottleneck (at reduced resolution)
        h_bot, c_bot = self.bottleneck_single.init_hidden(
            batch_size, height // 2, width // 2, device
        )
        
        # Process sequence
        for t in range(time_steps):
            current_input = x[:, t, :, :, :]
            
            # Downstream encoder
            h_down, c_down = self.downstream_encoder(current_input, (h_down, c_down))
            
            # Downsample
            encoded_feat = self.pool(h_down)
            
            # Bottleneck
            h_bot, c_bot = self.bottleneck_single(encoded_feat, (h_bot, c_bot))
            
            # Apply attention
            if self.use_attention:
                h_bot = self.attention(h_bot)
            
            # Apply dropout
            if self.dropout_rate > 0:
                h_bot = self.bottleneck_dropout(h_bot)
        
        # Decoder
        upsampled = torch.nn.functional.interpolate(
            h_bot,
            size=(h_down.size(2), h_down.size(3)),
            mode='bilinear',
            align_corners=True
        )
        
        concat_feat = torch.cat([upsampled, h_down], dim=1)
        
        h_dec, c_dec = self.decoder.init_hidden(batch_size, height, width, device)
        h_dec, c_dec = self.decoder(concat_feat, (h_dec, c_dec))
        
        # Output
        output = self.output_head(h_dec)
        
        return output
    
    def _forward_dual_stream(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for dual-stream (upstream + downstream).
        
        Args:
            x: Dict with keys:
               - 'downstream': [B, T, C, H_down, W_down]
               - 'upstream': [B, T, C, H_up, W_up]
        
        Returns:
            Predicted precipitation [B, 1, H_down, W_down]
        """
        downstream_input = x['downstream']
        upstream_input = x['upstream']
        
        batch_size = downstream_input.size(0)
        time_steps = downstream_input.size(1)
        device = downstream_input.device
        
        # Get spatial dimensions
        _, _, _, h_down, w_down = downstream_input.size()
        _, _, _, h_up, w_up = upstream_input.size()
        
        # Initialize downstream encoder
        h_down_enc, c_down_enc = self.downstream_encoder.init_hidden(
            batch_size, h_down, w_down, device
        )
        
        # Initialize upstream encoder
        h_up_enc, c_up_enc = self.upstream_encoder.init_hidden(
            batch_size, h_up, w_up, device
        )
        
        # Calculate bottleneck spatial dimensions
        # Use downstream dimensions as reference for fusion
        h_bot = h_down // 2
        w_bot = w_down // 2
        
        # Initialize bottleneck (dual-stream mode)
        h_bot_state, c_bot_state = self.bottleneck_dual.init_hidden(
            batch_size, h_bot, w_bot, device
        )
        
        # Process sequence through both streams
        for t in range(time_steps):
            # Get current timestep inputs
            down_current = downstream_input[:, t, :, :, :]
            up_current = upstream_input[:, t, :, :, :]
            
            # Process downstream encoder
            h_down_enc, c_down_enc = self.downstream_encoder(
                down_current, (h_down_enc, c_down_enc)
            )
            
            # Process upstream encoder
            h_up_enc, c_up_enc = self.upstream_encoder(
                up_current, (h_up_enc, c_up_enc)
            )
            
            # Downsample both streams
            down_encoded = self.pool(h_down_enc)  # [B, C, h_bot, w_bot]
            up_encoded = self.pool(h_up_enc)      # [B, C, h_up//2, w_up//2]
            
            # Resize upstream features to match downstream bottleneck size
            # This allows fusion even when regions have different dimensions
            up_encoded_resized = torch.nn.functional.interpolate(
                up_encoded,
                size=(h_bot, w_bot),
                mode='bilinear',
                align_corners=True
            )
            
            # Fuse features by concatenation
            fused_features = torch.cat([down_encoded, up_encoded_resized], dim=1)
            
            # Process through bottleneck
            h_bot_state, c_bot_state = self.bottleneck_dual(
                fused_features, (h_bot_state, c_bot_state)
            )
            
            # Apply attention
            if self.use_attention:
                h_bot_state = self.attention(h_bot_state)
            
            # Apply dropout
            if self.dropout_rate > 0:
                h_bot_state = self.bottleneck_dropout(h_bot_state)
        
        # Decoder (reconstruct downstream region only)
        upsampled = torch.nn.functional.interpolate(
            h_bot_state,
            size=(h_down_enc.size(2), h_down_enc.size(3)),
            mode='bilinear',
            align_corners=True
        )
        
        # Skip connection from downstream encoder
        concat_feat = torch.cat([upsampled, h_down_enc], dim=1)
        
        h_dec, c_dec = self.decoder.init_hidden(batch_size, h_down, w_down, device)
        h_dec, c_dec = self.decoder(concat_feat, (h_dec, c_dec))
        
        # Output
        output = self.output_head(h_dec)
        
        return output
