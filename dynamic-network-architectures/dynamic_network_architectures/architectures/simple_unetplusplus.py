from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class SimpleConvBlock(nn.Module):
    """Simple conv block"""
    def __init__(self, in_channels, out_channels, conv_op, norm_op, nonlin):
        super().__init__()
        self.conv = conv_op(in_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.norm = norm_op(out_channels, eps=1e-5, affine=True)
        self.nonlin = nonlin(negative_slope=1e-2, inplace=True)
    
    def forward(self, x):
        return self.nonlin(self.norm(self.conv(x)))


class StackedConvs(nn.Module):
    """Stacked conv blocks"""
    def __init__(self, in_channels, out_channels, num_convs, conv_op, norm_op, nonlin):
        super().__init__()
        blocks = []
        blocks.append(SimpleConvBlock(in_channels, out_channels, conv_op, norm_op, nonlin))
        for _ in range(num_convs - 1):
            blocks.append(SimpleConvBlock(out_channels, out_channels, conv_op, norm_op, nonlin))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.blocks(x)


class MinimalUNetPlusPlus(AbstractDynamicNetworkArchitectures):
    """
    Minimal UNet++ with only the most essential connections to reduce memory usage
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
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int], None] = None,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: Union[dict, None] = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: Union[dict, None] = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: Union[dict, None] = None,
        deep_supervision: bool = True,
        nonlin_first: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Process parameters
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        
        # Store key parameters
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.conv_op = conv_op
        self.norm_op = norm_op or nn.InstanceNorm3d
        self.nonlin = nonlin or nn.LeakyReLU
        self.use_checkpointing = True  # Enable gradient checkpointing
        
        # Build encoder - NO pooling, all same spatial dimensions
        self.conv_blocks_context = nn.ModuleList()
        input_features = input_channels
        for i in range(n_stages):
            self.conv_blocks_context.append(
                StackedConvs(input_features, features_per_stage[i], 2, conv_op, self.norm_op, self.nonlin)
            )
            input_features = features_per_stage[i]
        
        # Minimal UNet++ structure - only key connections
        # Only implement x0_1 and x0_2 to get the essence of UNet++
        
        # For x0_1: combine x0_0 and upsampled x1_0
        self.up1_0 = conv_op(features_per_stage[1], features_per_stage[0], kernel_size=1, padding=0, bias=True)
        self.conv_0_1 = StackedConvs(features_per_stage[0] * 2, features_per_stage[0], 2, conv_op, self.norm_op, self.nonlin)
        
        # For x0_2: combine x0_0, x0_1, and upsampled x1_1
        # First need x1_1: combine x1_0 and upsampled x2_0
        self.up2_0 = conv_op(features_per_stage[2], features_per_stage[1], kernel_size=1, padding=0, bias=True)
        self.conv_1_1 = StackedConvs(features_per_stage[1] * 2, features_per_stage[1], 2, conv_op, self.norm_op, self.nonlin)
        
        self.up1_1 = conv_op(features_per_stage[1], features_per_stage[0], kernel_size=1, padding=0, bias=True)
        self.conv_0_2 = StackedConvs(features_per_stage[0] * 3, features_per_stage[0], 2, conv_op, self.norm_op, self.nonlin)
        
        # Output layers - fewer outputs to reduce memory
        self.seg_outputs = nn.ModuleList([
            conv_op(features_per_stage[0], num_classes, kernel_size=1, bias=True),
            conv_op(features_per_stage[0], num_classes, kernel_size=1, bias=True)
        ])
        
        # Initialize weights
        self.apply(InitWeights_He(1e-2))
        
        # Decoder proxy for nnUNet compatibility
        class DecoderProxy:
            def __init__(self, parent):
                self.parent = parent
            
            @property
            def deep_supervision(self):
                return self.parent.deep_supervision
            
            @deep_supervision.setter
            def deep_supervision(self, value):
                self.parent.deep_supervision = value
        
        self.decoder = DecoderProxy(self)

    def _checkpoint_forward(self, func, *args):
        """Use gradient checkpointing if enabled"""
        if self.use_checkpointing and self.training:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def forward(self, x):
        """Forward pass with minimal UNet++ connections and checkpointing"""
        # Encoder - all same spatial dimensions
        x0_0 = self._checkpoint_forward(self.conv_blocks_context[0], x)    # 32 channels
        x1_0 = self._checkpoint_forward(self.conv_blocks_context[1], x0_0) # 64 channels  
        x2_0 = self._checkpoint_forward(self.conv_blocks_context[2], x1_0) # 128 channels
        
        # Minimal UNet++ connections
        # x0_1: combine x0_0 and upsampled x1_0
        up1_0 = nn.functional.interpolate(self.up1_0(x1_0), size=x0_0.shape[2:], mode='trilinear', align_corners=False)
        x0_1 = self._checkpoint_forward(self.conv_0_1, torch.cat([x0_0, up1_0], 1))  # [32, 32] -> 64 -> 32
        
        # x1_1: combine x1_0 and upsampled x2_0
        up2_0 = nn.functional.interpolate(self.up2_0(x2_0), size=x1_0.shape[2:], mode='trilinear', align_corners=False)
        x1_1 = self._checkpoint_forward(self.conv_1_1, torch.cat([x1_0, up2_0], 1))  # [64, 64] -> 128 -> 64
        
        # x0_2: combine x0_0, x0_1, and upsampled x1_1
        up1_1 = nn.functional.interpolate(self.up1_1(x1_1), size=x0_0.shape[2:], mode='trilinear', align_corners=False)
        x0_2 = self._checkpoint_forward(self.conv_0_2, torch.cat([x0_0, x0_1, up1_1], 1))  # [32, 32, 32] -> 96 -> 32
        
        # Apply segmentation heads - ensure output matches input spatial dimensions
        outputs = [
            nn.functional.interpolate(self.seg_outputs[0](x0_1), size=x.shape[2:], mode='trilinear', align_corners=False),
            nn.functional.interpolate(self.seg_outputs[1](x0_2), size=x.shape[2:], mode='trilinear', align_corners=False)
        ]
        
        if self.deep_supervision:
            return tuple(outputs[::-1])
        else:
            return outputs[-1]

    def compute_conv_feature_map_size(self, input_size):
        """Compute memory usage - much lower than full UNet++"""
        return int(np.prod(input_size) * 32 * 2)  # Reduced estimate

    @staticmethod  
    def initialize(module):
        InitWeights_He(1e-2)(module)


# Keep the old class name for compatibility
SimpleUNetPlusPlus = MinimalUNetPlusPlus 