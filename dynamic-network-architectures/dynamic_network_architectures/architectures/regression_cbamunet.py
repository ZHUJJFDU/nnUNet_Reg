from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.architectures.cbamunet import CBAM, CBAMPlainConvUNet


class RegcbamPlainConvUNet(PlainConvUNet):
    """
    Regression CBAM U-Net architecture that extends PlainConvUNet and incorporates CBAM attention.
    This network adds a regression branch to predict continuous values alongside segmentation.
    """
    
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        cbam_reduction_ratio: int = 16,
        cbam_spatial_kernel_size: int = 7,
        regression_dim: int = 1  # Number of regression values to predict
    ):
        """
        Initialize RegcbamPlainConvUNet
        
        Parameters are the same as PlainConvUNet with additional parameters:
        cbam_reduction_ratio: Channel reduction ratio for attention module
        cbam_spatial_kernel_size: Kernel size for spatial attention
        regression_dim: Dimension of the regression output (default: 1 for wall thickness)
        """
        # Initialize parent class (PlainConvUNet)
        super().__init__(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first
        )
        
        # Create CBAM modules
        self.cbam_modules = nn.ModuleList()
        
        # Get convolution dimension
        conv_dim = convert_conv_op_to_dim(conv_op)
        
        # Create CBAM modules for each stage
        if isinstance(features_per_stage, (list, tuple)):
            for feature_num in features_per_stage:
                cbam = CBAM(feature_num, cbam_reduction_ratio, cbam_spatial_kernel_size)
                cbam.set_conv_dim(conv_dim)
                self.cbam_modules.append(cbam)
        else:
            # If features_per_stage is a single integer
            for _ in range(n_stages):
                cbam = CBAM(features_per_stage, cbam_reduction_ratio, cbam_spatial_kernel_size)
                cbam.set_conv_dim(conv_dim)
                self.cbam_modules.append(cbam)
        
        # Create regression branch
        # Use the bottleneck features (deepest encoder features) for regression
        if isinstance(features_per_stage, (list, tuple)):
            bottleneck_features = features_per_stage[-1]
        else:
            bottleneck_features = features_per_stage
        
        # Regression branch consists of:
        # 1. Global average pooling to get a feature vector
        # 2. Two fully connected layers with ReLU activation
        # 3. Final layer to output regression values
        
        # Create global pooling based on convolution dimension
        if conv_dim == 2:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:  # conv_dim == 3
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            
        # Create regression branch
        self.regression_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, regression_dim)
        )
        
        # Initialize weights
        self.initialize_regression_branch()
        
    def initialize_regression_branch(self):
        """Initialize the weights of the regression branch."""
        for m in self.regression_branch.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with both segmentation and regression outputs
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: (segmentation_output, regression_output)
        """
        # Get encoder features
        skips = self.encoder(x)
        
        # Apply CBAM to enhance features
        enhanced_skips = []
        for i, skip in enumerate(skips):
            if i < len(self.cbam_modules):
                enhanced_skip = self.cbam_modules[i](skip)
                enhanced_skips.append(enhanced_skip)
            else:
                enhanced_skips.append(skip)
        
        # Get bottleneck features for regression (deepest encoder features)
        bottleneck_features = enhanced_skips[-1]
        
        # Get segmentation output
        seg_output = self.decoder(enhanced_skips)
        
        # Compute regression output
        pooled_features = self.global_pool(bottleneck_features)
        regression_output = self.regression_branch(pooled_features)
        
        # Return both outputs
        if isinstance(seg_output, tuple):  # Handle deep supervision case
            return (*seg_output, regression_output)
        else:
            return seg_output, regression_output
            
    @staticmethod
    def initialize(module):
        """Initialize network weights"""
        InitWeights_He(1e-2)(module) 