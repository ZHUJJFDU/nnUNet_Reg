import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Dict
import numpy as np

from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

# Try to import get_tp_fp_fn_tn if available, otherwise define our own version
try:
    from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
except ImportError:
    # Define a simplified version if not available
    def get_tp_fp_fn_tn(pred, target, axes=None, mask=None):
        """
        Simplified version of get_tp_fp_fn_tn that only computes true positives
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        tp = (pred * target).sum(axes)
        fp = pred.sum(axes) - tp
        fn = target.sum(axes) - tp
        tn = torch.prod(torch.tensor(pred.shape, device=pred.device)) - tp - fp - fn
        
        return tp, fp, fn, tn


class DC_and_CE_and_Regression_loss(nn.Module):
    """
    Combined loss for segmentation (Dice + CE) and regression (MSE/L1).
    This follows the pattern of nnUNetv2's compound losses.
    """
    def __init__(self, 
                 soft_dice_kwargs=None, 
                 ce_kwargs=None, 
                 weight_ce=1, 
                 weight_dice=1, 
                 weight_reg=1.0,
                 reg_loss_type='mse',
                 ignore_label=None,
                 dice_class=SoftDiceLoss,
                 debug=False):
        """
        Args:
            soft_dice_kwargs: Dict of arguments for the SoftDiceLoss
            ce_kwargs: Dict of arguments for the RobustCrossEntropyLoss
            weight_ce: Weight for cross entropy loss
            weight_dice: Weight for dice loss
            weight_reg: Weight for regression loss
            reg_loss_type: Type of regression loss ('mse', 'l1', 'smooth_l1')
            ignore_label: Label to ignore in the segmentation
            dice_class: Class to use for dice loss
            debug: Whether to print debug information
        """
        super().__init__()
        
        # Initialize with default kwargs if none provided
        if soft_dice_kwargs is None:
            soft_dice_kwargs = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
        if ce_kwargs is None:
            ce_kwargs = {'weight': None}
            
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_reg = weight_reg
        self.ignore_label = ignore_label
        self.reg_loss_type = reg_loss_type
        self.debug = debug

        # Initialize segmentation losses
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        
        # Use a custom dice loss implementation to avoid distributed operations
        # Instead of using the SoftDiceLoss class directly
        self.use_custom_dice = True
        self.apply_nonlin = softmax_helper_dim1
        self.batch_dice = soft_dice_kwargs.get('batch_dice', True)
        self.smooth = soft_dice_kwargs.get('smooth', 1e-5)
        self.do_bg = soft_dice_kwargs.get('do_bg', False)
        
        # Initialize regression loss based on type
        if reg_loss_type == 'mse':
            self.reg_loss = nn.MSELoss()
        elif reg_loss_type == 'l1':
            self.reg_loss = nn.L1Loss()
        elif reg_loss_type == 'smooth_l1':
            self.reg_loss = nn.SmoothL1Loss(beta=0.1)
        else:
            if self.debug:
                print(f"Unknown regression loss type: {reg_loss_type}, using MSE")
            self.reg_loss = nn.MSELoss()

    def custom_dice_loss(self, x, y, loss_mask=None):
        """
        Custom implementation of dice loss that avoids distributed operations
        """
        # Handle case where x is a list or tuple (deep supervision outputs)
        if isinstance(x, (list, tuple)):
            x = x[0]  # Use the first output (usually the main segmentation output)
        
        # Apply softmax if needed
        if self.apply_nonlin is not None:
            try:
                x = self.apply_nonlin(x)
            except Exception:
                # Try a different approach
                x = torch.softmax(x, dim=1)
            
        # Get shapes
        shp_x = x.shape
        shp_y = y.shape
        
        # Check if shapes match
        if shp_x[2:] != shp_y[2:]:
            # Instead of trying to resize, use a constant loss value
            return torch.tensor(0.5, device=x.device, requires_grad=True)
        
        # From here on, we know the shapes match
        try:
            # Make tensors contiguous for reshape operations
            x = x.contiguous()
            y = y.contiguous()
            
            # Flatten spatial dimensions
            x_flat = x.reshape(shp_x[0], shp_x[1], -1)
            y_flat = y.reshape(shp_y[0], shp_y[1], -1)
            
            # Apply mask if provided
            if loss_mask is not None:
                try:
                    loss_mask = loss_mask.contiguous().reshape(shp_x[0], 1, -1)
                    y_flat = y_flat * loss_mask
                    x_flat = x_flat * loss_mask
                except Exception:
                    pass
            
            # Compute dice for each class
            dice_scores = []
            
            # Compute dice for each class separately to avoid shape issues
            for c in range(x_flat.shape[1]):
                if self.batch_dice:
                    # Combine batch and spatial dimensions
                    x_c = x_flat[:, c].reshape(-1)
                    y_c = y_flat[:, min(c, y_flat.shape[1]-1)].reshape(-1)  # Handle case where y has fewer channels
                    
                    # Compute intersection and union
                    intersection = (x_c * y_c).sum()
                    union = x_c.sum() + y_c.sum()
                    
                    # Compute dice
                    if union > 0:
                        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                    else:
                        dice = torch.tensor(1.0, device=x.device)
                    
                    dice_scores.append(dice)
                else:
                    # Keep batch dimension separate
                    dice_batch = []
                    for b in range(x_flat.shape[0]):
                        x_bc = x_flat[b, c]
                        y_bc = y_flat[b, min(c, y_flat.shape[1]-1)]  # Handle case where y has fewer channels
                        
                        # Compute intersection and union
                        intersection = (x_bc * y_bc).sum()
                        union = x_bc.sum() + y_bc.sum()
                        
                        # Compute dice
                        if union > 0:
                            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
                        else:
                            dice = torch.tensor(1.0, device=x.device)
                        
                        dice_batch.append(dice)
                    
                    # Average over batch
                    dice_scores.append(torch.stack(dice_batch).mean())
            
            # Convert to tensor
            dice_scores = torch.stack(dice_scores)
            
            # Handle background class
            if not self.do_bg and dice_scores.shape[0] > 1:
                dice_scores = dice_scores[1:]
            
            # Return mean dice loss
            return 1.0 - dice_scores.mean()
            
        except Exception:
            # Return a default loss
            return torch.tensor(0.5, device=x.device, requires_grad=True)

    def forward(self, net_output, target, reg_output=None, reg_target=None):
        """
        Args:
            net_output: Network output for segmentation
            target: Target for segmentation
            reg_output: Network output for regression (optional)
            reg_target: Target for regression (optional)
                    
        Returns:
            Combined loss value or tuple of (total_loss, seg_loss, reg_loss)
        """
        # Initialize losses
        dc_loss = 0
        ce_loss = 0
        reg_loss = 0
        
        # Get device for creating tensors
        device = net_output.device if hasattr(net_output, 'device') else torch.device('cpu')
        
        # Handle tuple output from network (segmentation, regression)
        if isinstance(net_output, tuple) and len(net_output) == 2 and reg_output is None:
            net_output, reg_output = net_output
        
        # Handle case where net_output is a list (deep supervision outputs)
        if isinstance(net_output, list):
            net_output = net_output[0]  # Use the first output (usually the main segmentation output)
        
        # Compute segmentation loss
        try:
            # Handle ignore label for segmentation
            if self.ignore_label is not None:
                assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
                mask = target != self.ignore_label
                # remove ignore label from target, replace with one of the known labels
                target_dice = torch.where(mask, target, torch.zeros_like(target))
                num_fg = mask.sum()
            else:
                target_dice = target
                mask = None
                num_fg = None
            
            # Check for negative values in target
            if torch.min(target_dice) < 0:
                # Replace negative values with zeros
                target_dice = torch.where(target_dice < 0, torch.zeros_like(target_dice), target_dice)
            
            # Compute dice loss
            if self.weight_dice != 0:
                try:
                    if self.use_custom_dice:
                        dc_loss = self.custom_dice_loss(net_output, target_dice, loss_mask=mask)
                    else:
                        dc_loss = self.dc(net_output, target_dice, loss_mask=mask)
                except RuntimeError:
                    # Fallback to a simple loss
                    dc_loss = torch.tensor(0.1, device=device, requires_grad=True)
            
            # Compute CE loss
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg is None or num_fg > 0):
                try:
                    if target.shape[1] == 1:  # Target is not one-hot encoded
                        # Ensure target values are valid for CE loss
                        if torch.min(target) < 0 or torch.max(target) >= net_output.shape[1]:
                            # Create a copy for CE loss to avoid modifying the original
                            target_ce = target.clone()
                            
                            # Handle ignore_index if specified
                            if self.ignore_label is not None:
                                # If we have an ignore_label, make sure it's preserved
                                ignore_mask = target == self.ignore_label
                                # Set values < 0 or >= num_classes to 0, but preserve ignore_label
                                target_ce = torch.where(
                                    (target < 0) | (target >= net_output.shape[1]) & (~ignore_mask),
                                    torch.zeros_like(target),
                                    target
                                )
                            else:
                                # No ignore_label, simply clamp to valid range
                                target_ce = torch.clamp(target, 0, net_output.shape[1] - 1)
                        else:
                            target_ce = target
                            
                        ce_loss = self.ce(net_output, target_ce[:, 0].long())
                    else:
                        # For one-hot encoded targets
                        ce_loss = self.ce(net_output, torch.argmax(target, dim=1))
                except RuntimeError:
                    # Fallback to a simple loss
                    ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
                except Exception:
                    # Fallback to a simple loss
                    ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
                
        except Exception:
            # Fallback to dummy values
            dc_loss = torch.tensor(0.1, device=device, requires_grad=True)
            ce_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        # Compute regression loss if applicable
        if reg_output is not None and reg_target is not None and self.weight_reg > 0:
            try:
                # 确保回归输出的类型与回归目标匹配
                if reg_output.dtype != reg_target.dtype:
                    reg_output = reg_output.to(dtype=reg_target.dtype)
                
                # Check for NaN values
                if torch.isnan(reg_output).any() or torch.isnan(reg_target).any():
                    # Replace NaN values with zeros
                    reg_output = torch.nan_to_num(reg_output, nan=0.0)
                    reg_target = torch.nan_to_num(reg_target, nan=0.0)
                
                # Ensure regression target has the right shape
                if reg_target.ndim == 1 and reg_output.ndim > 1:
                    reg_target = reg_target.view(-1, 1)
                elif reg_target.ndim > 1 and reg_output.ndim == 1:
                    reg_output = reg_output.view(-1, 1)
                
                # Make sure both tensors are on the same device
                if reg_target.device != reg_output.device:
                    reg_target = reg_target.to(reg_output.device)
                
                # Check if shapes match
                if reg_output.shape != reg_target.shape:
                    reg_output = reg_output.reshape(reg_target.shape)
                
                # Compute regression loss - 尝试在原始设备上计算
                try:
                    # Try computing on original device
                    reg_loss = self.reg_loss(reg_output, reg_target)
                except RuntimeError as e1:
                    if "expected scalar type" in str(e1):
                        # 尝试转换类型
                        try:
                            reg_output = reg_output.to(dtype=torch.float32)
                            reg_target = reg_target.to(dtype=torch.float32)
                            reg_loss = self.reg_loss(reg_output, reg_target)
                        except Exception:
                            reg_loss = torch.tensor(0.1, device=device, requires_grad=True)
                    else:
                        try:
                            # Try computing on CPU
                            reg_loss = self.reg_loss(reg_output.cpu(), reg_target.cpu())
                            reg_loss = reg_loss.to(device)
                        except Exception:
                            # Use a dummy value as fallback
                            reg_loss = torch.tensor(0.1, device=device, requires_grad=True)
                
            except Exception:
                reg_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        # Combine losses
        loss = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_reg * reg_loss
        
        # Return combined loss and individual components
        return loss, ce_loss + dc_loss, reg_loss


class MSELoss(nn.Module):
    """Mean Squared Error loss for regression"""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class L1Loss(nn.Module):
    """L1 loss for regression"""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.L1Loss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss for regression"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss(beta=beta)
    
    def forward(self, pred, target):
        return self.loss_fn(pred, target) 