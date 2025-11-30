"""Deep dual-stream ConvLSTM U-Net for multi-region weather prediction.

This module implements a deeper dual-stream architecture with multiple encoder/decoder levels.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Union
from convlstm.model import ConvLSTMCell, SelfAttention


class DeepDualStreamConvLSTMUNet(nn.Module):
    """Deep dual-stream ConvLSTM U-Net with multiple encoder/decoder levels.
    
    This architecture extends the dual-stream concept to deeper networks,
    processing upstream and downstream regions through separate multi-level
    encoders and fusing at the bottleneck.
    """
    
    def __init__(self, 
                 input_channels: int = 56,
                 hidden_channels: List[int] = None,
                 output_channels: int = 1,
                 kernel_size: int = 3,
                 use_attention: bool = True,
                 use_group_norm: bool = True,
                 dropout_rate: float = 0.2):
        """Initialize DeepDualStreamConvLSTMUNet.
        
        Args:
            input_channels: Number of input channels (default: 56)
            hidden_channels: List of hidden dims [enc1, enc2, enc3, bottleneck] (default: [64, 128, 256, 512])
            output_channels: Number of output channels (default: 1)
            kernel_size: Convolutional kernel size (default: 3)
            use_attention: Whether to use self-attention at bottleneck (default: True)
            use_group_norm: Whether to use group normalization (default: True)
            dropout_rate: Dropout probability (default: 0.2)
        """
        super(DeepDualStreamConvLSTMUNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]
        
        if len(hidden_channels) < 3:
            raise ValueError("hidden_channels must contain at least 3 values")
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.num_levels = len(hidden_channels)
        self.use_attention = use_attention
        self.use_group_norm = use_group_norm
        self.dropout_rate = dropout_rate
        
        # Downstream encoders
        self.downstream_encoders = nn.ModuleList()
        self.downstream_encoders.append(
            ConvLSTMCell(input_channels, hidden_channels[0], kernel_size, use_group_norm=use_group_norm)
        )
        for i in range(1, self.num_levels - 1):
            self.downstream_encoders.append(
                ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size, use_group_norm=use_group_norm)
            )
        
        # Upstream encoders (same structure as downstream)
        self.upstream_encoders = nn.ModuleList()
        self.upstream_encoders.append(
            ConvLSTMCell(input_channels, hidden_channels[0], kernel_size, use_group_norm=use_group_norm)
        )
        for i in range(1, self.num_levels - 1):
            self.upstream_encoders.append(
                ConvLSTMCell(hidden_channels[i-1], hidden_channels[i], kernel_size, use_group_norm=use_group_norm)
            )
        
        # Pooling layers
        self.pools = nn.ModuleList([nn.MaxPool2d(2, 2) for _ in range(self.num_levels - 1)])
        
        # Dropout for encoders
        if dropout_rate > 0:
            self.encoder_dropouts = nn.ModuleList([
                nn.Dropout2d(dropout_rate) for _ in range(self.num_levels - 1)
            ])
        
        # Bottleneck (fuses both streams)
        # Input: concatenated features from both streams
        self.bottleneck_single = ConvLSTMCell(
            hidden_channels[-2], hidden_channels[-1], kernel_size, use_group_norm=use_group_norm
        )
        self.bottleneck_dual = ConvLSTMCell(
            2 * hidden_channels[-2], hidden_channels[-1], kernel_size, use_group_norm=use_group_norm
        )
        
        # Self-attention at bottleneck
        if use_attention:
            self.attention = SelfAttention(in_channels=hidden_channels[-1])
        
        # Bottleneck dropout
        if dropout_rate > 0:
            self.bottleneck_dropout = nn.Dropout2d(dropout_rate * 1.5)
        
        # Decoders (reconstruct downstream region)
        self.decoders = nn.ModuleList()
        for i in range(self.num_levels - 2, -1, -1):
            decoder_input_dim = hidden_channels[i+1] + hidden_channels[i]
            self.decoders.append(
                ConvLSTMCell(decoder_input_dim, hidden_channels[i], kernel_size, use_group_norm=use_group_norm)
            )
        
        # Decoder dropout
        if dropout_rate > 0:
            self.decoder_dropouts = nn.ModuleList([
                nn.Dropout2d(dropout_rate * 0.5) for _ in range(self.num_levels - 1)
            ])
        
        # Output head
        self.output_head = nn.Conv2d(hidden_channels[0], output_channels, kernel_size=1)
    
    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input data (single tensor or dict with 'downstream' and 'upstream')
        
        Returns:
            Predicted precipitation [B, 1, H_down, W_down]
        """
        is_dual_stream = isinstance(x, dict)
        
        if is_dual_stream:
            return self._forward_dual_stream(x)
        else:
            return self._forward_single_stream(x)
    
    def _forward_single_stream(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single-stream mode."""
        batch_size, time_steps, channels, height, width = x.size()
        device = x.device
        
        # Initialize encoder states
        encoder_states = []
        encoder_outputs = []
        
        h, w = height, width
        for encoder in self.downstream_encoders:
            h_enc, c_enc = encoder.init_hidden(batch_size, h, w, device)
            encoder_states.append((h_enc, c_enc))
            encoder_outputs.append(None)
            h, w = h // 2, w // 2
        
        # Initialize bottleneck
        h_bot, c_bot = self.bottleneck_single.init_hidden(batch_size, h, w, device)
        
        # Process sequence
        for t in range(time_steps):
            current_input = x[:, t, :, :, :]
            
            # Encoder forward
            for i, encoder in enumerate(self.downstream_encoders):
                h_enc, c_enc = encoder_states[i]
                h_enc, c_enc = encoder(current_input, (h_enc, c_enc))
                encoder_states[i] = (h_enc, c_enc)
                
                if self.dropout_rate > 0:
                    h_enc = self.encoder_dropouts[i](h_enc)
                
                encoder_outputs[i] = h_enc
                current_input = self.pools[i](h_enc)
            
            # Bottleneck
            h_bot, c_bot = self.bottleneck_single(current_input, (h_bot, c_bot))
            
            if self.use_attention:
                h_bot = self.attention(h_bot)
            
            if self.dropout_rate > 0:
                h_bot = self.bottleneck_dropout(h_bot)
        
        # Decoder
        decoder_input = h_bot
        
        for i, decoder in enumerate(self.decoders):
            encoder_idx = len(self.downstream_encoders) - 1 - i
            target_h, target_w = encoder_outputs[encoder_idx].size(2), encoder_outputs[encoder_idx].size(3)
            
            upsampled = torch.nn.functional.interpolate(
                decoder_input, size=(target_h, target_w), mode='bilinear', align_corners=True
            )
            
            concat_feat = torch.cat([upsampled, encoder_outputs[encoder_idx]], dim=1)
            
            h_dec, c_dec = decoder.init_hidden(batch_size, target_h, target_w, device)
            h_dec, c_dec = decoder(concat_feat, (h_dec, c_dec))
            
            if self.dropout_rate > 0:
                h_dec = self.decoder_dropouts[i](h_dec)
            
            decoder_input = h_dec
        
        output = self.output_head(decoder_input)
        return output
    
    def _forward_dual_stream(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for dual-stream mode."""
        downstream_input = x['downstream']
        upstream_input = x['upstream']
        
        batch_size = downstream_input.size(0)
        time_steps = downstream_input.size(1)
        device = downstream_input.device
        
        _, _, _, h_down, w_down = downstream_input.size()
        _, _, _, h_up, w_up = upstream_input.size()
        
        # Initialize downstream encoder states
        down_encoder_states = []
        down_encoder_outputs = []
        
        h, w = h_down, w_down
        for encoder in self.downstream_encoders:
            h_enc, c_enc = encoder.init_hidden(batch_size, h, w, device)
            down_encoder_states.append((h_enc, c_enc))
            down_encoder_outputs.append(None)
            h, w = h // 2, w // 2
        
        # Initialize upstream encoder states
        up_encoder_states = []
        
        h, w = h_up, w_up
        for encoder in self.upstream_encoders:
            h_enc, c_enc = encoder.init_hidden(batch_size, h, w, device)
            up_encoder_states.append((h_enc, c_enc))
            h, w = h // 2, w // 2
        
        # Bottleneck dimensions (based on downstream)
        h_bot = h_down
        w_bot = w_down
        for _ in range(self.num_levels - 1):
            h_bot, w_bot = h_bot // 2, w_bot // 2
        
        # Initialize bottleneck
        h_bot_state, c_bot_state = self.bottleneck_dual.init_hidden(batch_size, h_bot, w_bot, device)
        
        # Process sequence
        for t in range(time_steps):
            down_current = downstream_input[:, t, :, :, :]
            up_current = upstream_input[:, t, :, :, :]
            
            # Process downstream encoders
            down_input = down_current
            for i, encoder in enumerate(self.downstream_encoders):
                h_enc, c_enc = down_encoder_states[i]
                h_enc, c_enc = encoder(down_input, (h_enc, c_enc))
                down_encoder_states[i] = (h_enc, c_enc)
                
                if self.dropout_rate > 0:
                    h_enc = self.encoder_dropouts[i](h_enc)
                
                down_encoder_outputs[i] = h_enc
                down_input = self.pools[i](h_enc)
            
            # Process upstream encoders
            up_input = up_current
            for i, encoder in enumerate(self.upstream_encoders):
                h_enc, c_enc = up_encoder_states[i]
                h_enc, c_enc = encoder(up_input, (h_enc, c_enc))
                up_encoder_states[i] = (h_enc, c_enc)
                
                if self.dropout_rate > 0:
                    h_enc = self.encoder_dropouts[i](h_enc)
                
                up_input = self.pools[i](h_enc)
            
            # Resize upstream to match downstream bottleneck input size
            up_resized = torch.nn.functional.interpolate(
                up_input, size=(down_input.size(2), down_input.size(3)),
                mode='bilinear', align_corners=True
            )
            
            # Fuse features
            fused_features = torch.cat([down_input, up_resized], dim=1)
            
            # Bottleneck
            h_bot_state, c_bot_state = self.bottleneck_dual(fused_features, (h_bot_state, c_bot_state))
            
            if self.use_attention:
                h_bot_state = self.attention(h_bot_state)
            
            if self.dropout_rate > 0:
                h_bot_state = self.bottleneck_dropout(h_bot_state)
        
        # Decoder (reconstruct downstream only)
        decoder_input = h_bot_state
        
        for i, decoder in enumerate(self.decoders):
            encoder_idx = len(self.downstream_encoders) - 1 - i
            target_h = down_encoder_outputs[encoder_idx].size(2)
            target_w = down_encoder_outputs[encoder_idx].size(3)
            
            upsampled = torch.nn.functional.interpolate(
                decoder_input, size=(target_h, target_w), mode='bilinear', align_corners=True
            )
            
            concat_feat = torch.cat([upsampled, down_encoder_outputs[encoder_idx]], dim=1)
            
            h_dec, c_dec = decoder.init_hidden(batch_size, target_h, target_w, device)
            h_dec, c_dec = decoder(concat_feat, (h_dec, c_dec))
            
            if self.dropout_rate > 0:
                h_dec = self.decoder_dropouts[i](h_dec)
            
            decoder_input = h_dec
        
        output = self.output_head(decoder_input)
        return output
