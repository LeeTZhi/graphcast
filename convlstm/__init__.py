"""ConvLSTM-based weather prediction module.

This module implements a ConvLSTM U-Net architecture for regional precipitation
forecasting using spatio-temporal atmospheric data.
"""

from convlstm.config import ConvLSTMConfig
from convlstm.model import ConvLSTMCell, ConvLSTMUNet, WeightedPrecipitationLoss
from convlstm.data import (
    stack_channels, 
    ConvLSTMDataset, 
    RegionConfig, 
    ConvLSTMNormalizer,
    create_train_val_test_split
)
from convlstm.trainer import ConvLSTMTrainer

__all__ = [
    'ConvLSTMConfig',
    'ConvLSTMCell',
    'ConvLSTMUNet',
    'WeightedPrecipitationLoss',
    'ConvLSTMTrainer',
    'stack_channels',
    'ConvLSTMDataset',
    'RegionConfig',
    'ConvLSTMNormalizer',
    'create_train_val_test_split'
]
