"""Masked loss functions for handling missing/filled data in weather prediction.

These loss functions use validity masks to reduce the influence of filled/interpolated
values during training, allowing the model to focus on reliable observations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """MSE loss with validity mask support.
    
    This loss reduces the weight of filled/interpolated values, allowing
    the model to focus on reliable observations while still learning from
    the spatial structure of filled regions.
    """
    
    def __init__(self, 
                 filled_weight: float = 0.1,
                 reduction: str = 'mean'):
        """Initialize MaskedMSELoss.
        
        Args:
            filled_weight: Weight for filled/invalid values (0-1, default: 0.1)
                          0.0 = completely ignore filled values
                          1.0 = treat filled values same as valid values
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.filled_weight = filled_weight
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute masked MSE loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            mask: Validity mask [B, C, H, W] where 1=valid, 0=filled
                  If None, treats all values as valid
        
        Returns:
            Scalar loss value
        """
        # Compute squared error
        squared_error = (pred - target) ** 2
        
        if mask is None:
            # No mask provided, use standard MSE
            if self.reduction == 'mean':
                return squared_error.mean()
            elif self.reduction == 'sum':
                return squared_error.sum()
            else:
                return squared_error
        
        # Apply mask weights
        # Valid points: weight = 1.0
        # Filled points: weight = filled_weight
        weights = mask + (1 - mask) * self.filled_weight
        
        # Weighted squared error
        weighted_error = squared_error * weights
        
        if self.reduction == 'mean':
            # Normalize by sum of weights to maintain scale
            return weighted_error.sum() / (weights.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_error.sum()
        else:
            return weighted_error


class MaskedL1Loss(nn.Module):
    """L1 (MAE) loss with validity mask support."""
    
    def __init__(self, 
                 filled_weight: float = 0.1,
                 reduction: str = 'mean'):
        """Initialize MaskedL1Loss.
        
        Args:
            filled_weight: Weight for filled/invalid values (0-1, default: 0.1)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.filled_weight = filled_weight
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute masked L1 loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            mask: Validity mask [B, C, H, W] where 1=valid, 0=filled
        
        Returns:
            Scalar loss value
        """
        # Compute absolute error
        abs_error = torch.abs(pred - target)
        
        if mask is None:
            if self.reduction == 'mean':
                return abs_error.mean()
            elif self.reduction == 'sum':
                return abs_error.sum()
            else:
                return abs_error
        
        # Apply mask weights
        weights = mask + (1 - mask) * self.filled_weight
        weighted_error = abs_error * weights
        
        if self.reduction == 'mean':
            return weighted_error.sum() / (weights.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_error.sum()
        else:
            return weighted_error


class MaskedHuberLoss(nn.Module):
    """Huber loss with validity mask support.
    
    Huber loss is less sensitive to outliers than MSE, which is useful
    for weather data that may have extreme values.
    """
    
    def __init__(self, 
                 delta: float = 1.0,
                 filled_weight: float = 0.1,
                 reduction: str = 'mean'):
        """Initialize MaskedHuberLoss.
        
        Args:
            delta: Threshold for switching between L2 and L1 (default: 1.0)
            filled_weight: Weight for filled/invalid values (0-1, default: 0.1)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.delta = delta
        self.filled_weight = filled_weight
        self.reduction = reduction
    
    def forward(self, 
                pred: torch.Tensor, 
                target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """Compute masked Huber loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            mask: Validity mask [B, C, H, W] where 1=valid, 0=filled
        
        Returns:
            Scalar loss value
        """
        # Compute Huber loss
        error = pred - target
        abs_error = torch.abs(error)
        
        # Huber loss: L2 for small errors, L1 for large errors
        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=error.device))
        linear = abs_error - quadratic
        huber_loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        if mask is None:
            if self.reduction == 'mean':
                return huber_loss.mean()
            elif self.reduction == 'sum':
                return huber_loss.sum()
            else:
                return huber_loss
        
        # Apply mask weights
        weights = mask + (1 - mask) * self.filled_weight
        weighted_loss = huber_loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.sum() / (weights.sum() + 1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class CombinedMaskedLoss(nn.Module):
    """Combined loss with multiple components and mask support.
    
    Combines MSE, gradient loss, and optional perceptual loss with
    validity mask weighting.
    """
    
    def __init__(self,
                 mse_weight: float = 1.0,
                 gradient_weight: float = 0.1,
                 filled_weight: float = 0.1):
        """Initialize CombinedMaskedLoss.
        
        Args:
            mse_weight: Weight for MSE loss component
            gradient_weight: Weight for spatial gradient loss
            filled_weight: Weight for filled/invalid values (0-1)
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        self.mse_loss = MaskedMSELoss(filled_weight=filled_weight)
    
    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None) -> dict:
        """Compute combined masked loss.
        
        Args:
            pred: Predictions [B, C, H, W]
            target: Ground truth [B, C, H, W]
            mask: Validity mask [B, C, H, W] where 1=valid, 0=filled
        
        Returns:
            Dictionary with 'total' loss and individual components
        """
        losses = {}
        
        # MSE loss
        mse = self.mse_loss(pred, target, mask)
        losses['mse'] = mse
        
        # Gradient loss (spatial consistency)
        if self.gradient_weight > 0:
            grad_loss = self._gradient_loss(pred, target, mask)
            losses['gradient'] = grad_loss
        else:
            grad_loss = 0.0
        
        # Total loss
        total = self.mse_weight * mse + self.gradient_weight * grad_loss
        losses['total'] = total
        
        return losses
    
    def _gradient_loss(self,
                      pred: torch.Tensor,
                      target: torch.Tensor,
                      mask: torch.Tensor = None) -> torch.Tensor:
        """Compute spatial gradient loss.
        
        This encourages the model to preserve spatial patterns and gradients,
        which is important for weather fields.
        """
        # Compute gradients using Sobel-like filters
        # Horizontal gradient
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Vertical gradient
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Gradient differences
        grad_diff_x = (pred_dx - target_dx) ** 2
        grad_diff_y = (pred_dy - target_dy) ** 2
        
        if mask is not None:
            # Apply mask to gradients (use minimum of adjacent pixels)
            mask_x = torch.min(mask[:, :, :, 1:], mask[:, :, :, :-1])
            mask_y = torch.min(mask[:, :, 1:, :], mask[:, :, :-1, :])
            
            grad_diff_x = grad_diff_x * mask_x
            grad_diff_y = grad_diff_y * mask_y
            
            # Normalize by valid gradient count
            total_grad = grad_diff_x.sum() + grad_diff_y.sum()
            total_weight = mask_x.sum() + mask_y.sum()
            return total_grad / (total_weight + 1e-8)
        else:
            return (grad_diff_x.mean() + grad_diff_y.mean()) / 2
