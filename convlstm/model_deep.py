"""Deeper ConvLSTM U-Net model for better feature extraction.

This is an enhanced version with more encoder/decoder layers.
"""

import torch
import torch.nn as nn
from typing import List
from convlstm.model import ConvLSTMCell


class DeepConvLSTMUNet(nn.Module):
    """Deeper ConvLSTM U-Net with 3-4 encoder/decoder levels.
    
    Architecture:
        - Encoder: Multiple ConvLSTM layers with progressive downsampling
        - Bottleneck: ConvLSTM at lowest resolution
        - Decoder: Progressive upsampling with skip connections
        - Output: 1x1 conv for precipitation
    """
    
    def __init__(self, 
                 input_channels: int = 56,
                 hidden_channels: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: int = 3):
        """Initialize DeepConvLSTMUNet.
        
        Args:
            input_channels: Number of input channels (default: 56)
            hidden_channels: List of hidden dims [enc1, enc2, enc3, bottleneck] (default: [64, 128, 256, 512])
            output_channels: Number of output channels (default: 1)
            kernel_size: Convolutional kernel size (default: 3)
        """
        super(DeepConvLSTMUNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]
        
        if len(hidden_channels) < 3:
            raise ValueError("hidden_channels must contain at least 3 values")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_levels = len(hidden_channels)
        
        # Encoder layers
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvLSTMCell(input_channels, hidden_channels[0], kernel_size))
        for i in range(1, self.num_levels - 1):
            self.encoders.append(ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size))
        
        # Pooling layers
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(self.num_levels - 1)])
        
        # Bottleneck
        self.bottleneck = ConvLSTMCell(hidden_channels[-2], hidden_channels[-1], kernel_size)
        
        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            # Input: upsampled + skip connection
            decoder_input_dim = hidden_channels[i+1] + hidden_channels[i]
            self.decoders.append(ConvLSTMCell(decoder_input_dim, hidden_channels[i], kernel_size))
        
        # Output head
        self.output_head = nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input [B, T, C, H, W]
            
        Returns:
            Predicted precipitation [B, 1, H, W]
        """
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
            
            # Encoder forward pass
            for i, encoder in enumerate(self.encoders):
                h_enc, c_enc = encoder_states[i]
                h_enc, c_enc = encoder(current_input, (h_enc, c_enc))
                encoder_states[i] = (h_enc, c_enc)
                encoder_outputs[i] = h_enc
                
                # Downsample for next level
                # All encoders need pooling: encoder output -> pool -> next encoder (or bottleneck)
                current_input = self.pools[i](h_enc)
            
            # Bottleneck processes the downsampled output from last encoder
            h_bot, c_bot = self.bottleneck(current_input, (h_bot, c_bot))
        
        # Decoder with skip connections
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
            
            decoder_input = h_dec
        
        # Output
        output = self.output_head(decoder_input)
        
        return output
