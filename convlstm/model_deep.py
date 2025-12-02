"""Deeper ConvLSTM U-Net model for better feature extraction.

This is an enhanced version with more encoder/decoder layers.
"""

import torch
import torch.nn as nn
from typing import List
from convlstm.model import ConvLSTMCell, SelfAttention


class DeepConvLSTMUNet(nn.Module):
    """Deeper ConvLSTM U-Net with 3-4 encoder/decoder levels and regularization.
    
    Architecture:
        - Encoder: Multiple ConvLSTM layers with progressive downsampling
        - Bottleneck: ConvLSTM at lowest resolution
        - Decoder: Progressive upsampling with skip connections
        - Output: 1x1 conv for precipitation
        
    Regularization techniques:
        - Dropout: Applied after each encoder/decoder layer
        - Batch Normalization: Stabilizes training and acts as regularization
        - Spatial Dropout: Drops entire feature maps (better for spatial data)
    """
    
    def __init__(self, 
                 input_channels: int = 56,
                 hidden_channels: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: int = 3,
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = False,
                 use_group_norm: bool = True,
                 use_spatial_dropout: bool = True,
                 use_attention: bool = True,
                 multi_variable: bool = False):
        """Initialize DeepConvLSTMUNet with regularization.
        
        Args:
            input_channels: Number of input channels (default: 56)
            hidden_channels: List of hidden dims [enc1, enc2, enc3, bottleneck] (default: [64, 128, 256, 512])
            output_channels: Number of output channels (default: 1, ignored if multi_variable=True)
            kernel_size: Convolutional kernel size (default: 3)
            dropout_rate: Dropout probability (default: 0.2, set to 0 to disable)
            use_batch_norm: Whether to use batch normalization (default: False, not recommended)
            use_group_norm: Whether to use group normalization in ConvLSTM cells (default: True, recommended)
            use_spatial_dropout: Use spatial dropout instead of regular dropout (default: True)
            use_attention: Whether to use self-attention at bottleneck (default: True)
            multi_variable: Whether to predict all variables (56 channels) or just precipitation (1 channel) (default: False)
        """
        super(DeepConvLSTMUNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]
        
        if len(hidden_channels) < 3:
            raise ValueError("hidden_channels must contain at least 3 values")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.multi_variable = multi_variable
        self.output_channels = 56 if multi_variable else output_channels
        self.kernel_size = kernel_size
        self.num_levels = len(hidden_channels)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_group_norm = use_group_norm
        self.use_attention = use_attention
        
        # Encoder layers (with Group Normalization support)
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size, 
                                         use_group_norm=use_group_norm))
        for i in range(1, self.num_levels - 1):
            self.encoders.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size,
                                             use_group_norm=use_group_norm))
        
        # Encoder batch normalization (optional)
        if use_batch_norm:
            self.encoder_bns = nn.ModuleList([
                nn.BatchNorm2d(hidden_channels[i]) for i in range(self.num_levels - 1)
            ])
        
        # Encoder dropout (spatial or regular)
        if dropout_rate > 0:
            if use_spatial_dropout:
                self.encoder_dropouts = nn.ModuleList([
                    nn.Dropout2d(dropout_rate) for _ in range(self.num_levels - 1)
                ])
            else:
                self.encoder_dropouts = nn.ModuleList([
                    nn.Dropout(dropout_rate) for _ in range(self.num_levels - 1)
                ])
        
        # Pooling layers
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(self.num_levels - 1)])
        
        # Bottleneck (with Group Normalization support)
        self.bottleneck = ConvLSTMCell(hidden_channels[-2], hidden_channels[-1], kernel_size,
                                      use_group_norm=use_group_norm)
        
        # Self-Attention at bottleneck (captures global context at lowest resolution)
        if use_attention:
            self.attention = SelfAttention(in_channels=hidden_channels[-1])
        
        # Bottleneck batch norm and dropout
        if use_batch_norm:
            self.bottleneck_bn = nn.BatchNorm2d(hidden_channels[-1])
        if dropout_rate > 0:
            if use_spatial_dropout:
                self.bottleneck_dropout = nn.Dropout2d(dropout_rate * 1.5)  # Stronger dropout at bottleneck
            else:
                self.bottleneck_dropout = nn.Dropout(dropout_rate * 1.5)
        
        # Decoder layers (with Group Normalization support)
        self.decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            # Input: upsampled + skip connection
            decoder_input_dim = hidden_channels[i+1] + hidden_channels[i]
            self.decoders.append(ConvLSTMCell(decoder_input_dim, hidden_channels[i], kernel_size,
                                             use_group_norm=use_group_norm))
        
        # Decoder batch normalization (optional)
        if use_batch_norm:
            self.decoder_bns = nn.ModuleList([
                nn.BatchNorm2d(hidden_channels[i]) for i in range(self.num_levels - 2, -1, -1)
            ])
        
        # Decoder dropout
        if dropout_rate > 0:
            if use_spatial_dropout:
                self.decoder_dropouts = nn.ModuleList([
                    nn.Dropout2d(dropout_rate * 0.5) for _ in range(self.num_levels - 1)  # Lighter dropout in decoder
                ])
            else:
                self.decoder_dropouts = nn.ModuleList([
                    nn.Dropout(dropout_rate * 0.5) for _ in range(self.num_levels - 1)
                ])
        
        # Output head
        self.output_head = nn.Conv2d(hidden_channels[0], self.output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [B, T, C, H, W] or dict with 'downstream' key
            
        Returns:
            Predicted precipitation [B, 1, H, W]
            
        Note:
            If x is a dict (dual-stream mode), only uses 'downstream' key.
        """
        # Handle dict input (dual-stream mode) - use only downstream
        if isinstance(x, dict):
            x = x['downstream']
        
        batch_size, time_steps, channels, height, width = x.size()
        device = x.device
        
        # Initialize encoder hidden states
        # Each encoder operates at its own spatial resolution
        encoder_states = []
        encoder_outputs = []
        
        h, w = height, width
        for i, encoder in enumerate(self.encoders):
            h_enc, c_enc = encoder.init_hidden(batch_size, h, w, device)
            encoder_states.append((h_enc, c_enc))
            encoder_outputs.append(None)
            # Calculate size for next level (after pooling)
            h, w = h // 2, w // 2
        
        # Initialize bottleneck hidden state at the smallest resolution
        # After all encoder pooling operations
        h_bot, c_bot = self.bottleneck.init_hidden(batch_size, h, w, device)
        
        # Process sequence through encoders and bottleneck
        for t in range(time_steps):
            current_input = x[:, t, :, :, :]
            
            # Encoder forward pass with regularization
            for i, encoder in enumerate(self.encoders):
                h_enc, c_enc = encoder_states[i]
                h_enc, c_enc = encoder(current_input, (h_enc, c_enc))
                encoder_states[i] = (h_enc, c_enc)
                
                # Apply batch normalization
                if self.use_batch_norm:
                    h_enc = self.encoder_bns[i](h_enc)
                
                # Apply dropout
                if self.dropout_rate > 0:
                    h_enc = self.encoder_dropouts[i](h_enc)
                
                encoder_outputs[i] = h_enc
                
                # Downsample for next level
                current_input = self.pools[i](h_enc)
            
            # Bottleneck with regularization
            h_bot, c_bot = self.bottleneck(current_input, (h_bot, c_bot))
            
            # Apply Self-Attention at bottleneck (captures global context at each timestep)
            if self.use_attention:
                h_bot = self.attention(h_bot)
            
            # Apply batch normalization to bottleneck (if enabled, though not recommended)
            if self.use_batch_norm:
                h_bot = self.bottleneck_bn(h_bot)
            
            # Apply dropout to bottleneck
            if self.dropout_rate > 0:
                h_bot = self.bottleneck_dropout(h_bot)
        
        # Decoder with skip connections and regularization
        decoder_input = h_bot
        
        for i, decoder in enumerate(self.decoders):
            # Upsample
            encoder_idx = len(self.encoders) - 1 - i
            target_h, target_w = encoder_outputs[encoder_idx].size(2), encoder_outputs[encoder_idx].size(3)
            
            upsampled = torch.nn.functional.interpolate(
                decoder_input,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=True
            )
            
            # Skip connection
            concat_feat = torch.cat([upsampled, encoder_outputs[encoder_idx]], dim=1)
            
            # Decoder ConvLSTM
            h_dec, c_dec = decoder.init_hidden(batch_size, target_h, target_w, device)
            h_dec, c_dec = decoder(concat_feat, (h_dec, c_dec))
            
            # Apply batch normalization
            if self.use_batch_norm:
                h_dec = self.decoder_bns[i](h_dec)
            
            # Apply dropout
            if self.dropout_rate > 0:
                h_dec = self.decoder_dropouts[i](h_dec)
            
            decoder_input = h_dec
        
        # Output
        output = self.output_head(decoder_input)
        
        return output
