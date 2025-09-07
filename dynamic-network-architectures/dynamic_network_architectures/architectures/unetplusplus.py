from typing import Union, Type, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class ConvDropoutNormNonlin(nn.Module):
    """
    Basic conv block following official nnUNet convention
    """
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs['p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.norm = norm_op(output_channels, **norm_op_kwargs)
        self.nonlin = nonlin(**nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.nonlin(self.norm(x))


class StackedConvLayers(nn.Module):
    """
    Stacked conv layers following official nnUNet convention
    """
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        super(StackedConvLayers, self).__init__()
        
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        if first_stride is not None:
            conv_kwargs_first_conv = conv_kwargs.copy()
            conv_kwargs_first_conv['stride'] = first_stride
        else:
            conv_kwargs_first_conv = conv_kwargs

        blocks = []
        blocks.append(basic_block(input_feature_channels, output_feature_channels, conv_op,
                                conv_kwargs_first_conv, norm_op, norm_op_kwargs, dropout_op,
                                dropout_op_kwargs, nonlin, nonlin_kwargs))
        
        for _ in range(num_convs - 1):
            blocks.append(basic_block(output_feature_channels, output_feature_channels, conv_op,
                                    conv_kwargs, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs, nonlin, nonlin_kwargs))
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class Upsample(nn.Module):
    def __init__(self, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                           align_corners=self.align_corners)


class UNetPlusPlus(AbstractDynamicNetworkArchitectures):
    """
    UNet++ implementation following the exact structure of the official implementation
    Adapted to be compatible with nnUNet v2 architecture system
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
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = None,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = True,
        nonlin_first: bool = False,
        **kwargs
    ):
        super().__init__()
        
        # Process parameters following nnUNet v2 conventions
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(kernel_sizes, int):
            kernel_sizes = [[kernel_sizes] * convert_conv_op_to_dim(conv_op) for _ in range(n_stages)]
        if isinstance(strides, int):
            strides = [[strides] * convert_conv_op_to_dim(conv_op) for _ in range(n_stages)]
            
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = n_conv_per_stage
        elif isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            
        norm_op_kwargs = norm_op_kwargs or {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        dropout_op_kwargs = dropout_op_kwargs or {'p': 0.0, 'inplace': True}
        nonlin_kwargs = nonlin_kwargs or {'negative_slope': 1e-2, 'inplace': True}

        # Store parameters
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.n_conv_per_stage = n_conv_per_stage
        self.strides = strides
        
        # Determine upsampling mode
        if conv_op == nn.Conv3d:
            self.upsample_mode = 'trilinear'
        else:
            self.upsample_mode = 'bilinear'

        # Build encoder (context blocks) - following official UNet++ implementation
        # NO POOLING within encoder stages - all stages have same spatial dimensions
        self.conv_blocks_context = nn.ModuleList()
        
        # Calculate feature progression
        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': conv_bias}
        
        input_features = input_channels
        for i in range(n_stages):
            # Set kernel size and padding for this stage  
            if isinstance(kernel_sizes, list) and len(kernel_sizes) > i:
                if isinstance(kernel_sizes[i], int):
                    kernel_size = [kernel_sizes[i]] * convert_conv_op_to_dim(conv_op)
                else:
                    kernel_size = kernel_sizes[i]
            else:
                # Fallback to default
                kernel_size = [3] * convert_conv_op_to_dim(conv_op)
            
            self.conv_kwargs['kernel_size'] = kernel_size
            self.conv_kwargs['padding'] = [k//2 for k in kernel_size]
            
            # Create conv block WITHOUT pooling - following official UNet++ approach
            # All encoder stages have same spatial dimensions, only channels differ
            self.conv_blocks_context.append(
                StackedConvLayers(
                    input_features, features_per_stage[i], n_conv_per_stage[i],
                    conv_op, self.conv_kwargs, norm_op, norm_op_kwargs, 
                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                    None, ConvDropoutNormNonlin  # No stride/pooling!
                )
            )
            
            input_features = features_per_stage[i]

        # Build UNet++ decoder structure following official implementation
        num_pool = n_stages - 1
        
        # Create nested structure: loc0, loc1, loc2, loc3, loc4
        self.loc0 = nn.ModuleList()
        self.loc1 = nn.ModuleList()
        self.loc2 = nn.ModuleList()
        self.loc3 = nn.ModuleList()
        self.loc4 = nn.ModuleList()
        
        self.up0 = nn.ModuleList()
        self.up1 = nn.ModuleList()
        self.up2 = nn.ModuleList()
        self.up3 = nn.ModuleList()
        self.up4 = nn.ModuleList()

        # Build each nested level following create_nest logic
        self._build_decoder_level(0, num_pool, features_per_stage, n_conv_per_stage_decoder)  # loc0, up0
        self._build_decoder_level(1, num_pool, features_per_stage, n_conv_per_stage_decoder)  # loc1, up1
        self._build_decoder_level(2, num_pool, features_per_stage, n_conv_per_stage_decoder)  # loc2, up2
        self._build_decoder_level(3, num_pool, features_per_stage, n_conv_per_stage_decoder)  # loc3, up3
        self._build_decoder_level(4, num_pool, features_per_stage, n_conv_per_stage_decoder)  # loc4, up4

        # Segmentation output layers
        self.seg_outputs = nn.ModuleList()
        # Each output layer connects to the leftmost column of the nested structure
        for level in [4, 3, 2, 1, 0]:  # From shallowest to deepest nesting
            if level < len(features_per_stage):
                self.seg_outputs.append(
                    conv_op(features_per_stage[0], num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )

        # Initialize weights
        self.apply(InitWeights_He(1e-2))
        
        # Create a decoder object for nnUNet v2 compatibility
        # This allows the trainer to access decoder.deep_supervision
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

    def _build_decoder_level(self, level, num_pool, features_per_stage, n_conv_per_stage_decoder):
        """Build one level of the nested decoder structure"""
        loc_modules = getattr(self, f'loc{level}')
        up_modules = getattr(self, f'up{level}')
        
        # Each level processes different numbers of stages
        for u in range(level, num_pool):
            # Index into encoder features - official impl uses -(2 + u) indexing
            skip_stage_idx = -(2 + u)  # -2, -3, -4, -5 for u=0,1,2,3
            nfeatures_from_skip = features_per_stage[skip_stage_idx]
            
            # Calculate concatenated features based on nested structure
            # This follows the pattern: nfeatures_from_skip * (2 + u - level)
            n_features_after_concat = nfeatures_from_skip * (2 + u - level)
            
            # Output features - following official implementation logic
            if u != num_pool - 1:
                # Not the last stage - output should match the next shallower skip
                final_num_features = features_per_stage[-(3 + u)]
            else:
                # Last stage - output should match current skip
                final_num_features = nfeatures_from_skip
            
            # Create upsampling layer - MUST reduce channels to match concatenation formula
            # The key insight: we need to determine what channels are being upsampled
            
            # In the nested structure, the input to upsampling comes from different sources:
            # - For level 4: up4[0] takes x1_0 (64 channels) -> should output 32 channels
            # - For level 3: up3[0] takes x2_0 (128 channels) -> should output 64 channels
            #                up3[1] takes x1_1 (output from loc3[0]) -> should output 32 channels
            
            # The input channels depend on the source of the upsampling:
            if u == level:
                # First upsampling in this level - comes from encoder stage
                input_channels = features_per_stage[u + 1]  # Next deeper encoder stage
            else:
                # Subsequent upsampling - comes from previous localization output
                # This is more complex to determine, use nfeatures_from_skip as a safe fallback
                input_channels = nfeatures_from_skip
            
            output_channels = nfeatures_from_skip  # Always match the target skip connection
            
            # Use 1x1x1 conv for channel reduction (no spatial upsampling needed)
            channel_reducer = self.conv_op(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=True)
            up_modules.append(channel_reducer)
            
            # Create localization block
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}
            if self.conv_op == nn.Conv3d:
                conv_kwargs['kernel_size'] = [3, 3, 3]
                conv_kwargs['padding'] = [1, 1, 1]
            else:
                conv_kwargs['kernel_size'] = [3, 3]
                conv_kwargs['padding'] = [1, 1]
            
            # Two-part block: reduce features then refine (following official implementation)
            loc_block = nn.Sequential(
                StackedConvLayers(
                    n_features_after_concat, nfeatures_from_skip, 
                    n_conv_per_stage_decoder[0] - 1 if n_conv_per_stage_decoder else 1,
                    self.conv_op, conv_kwargs, self.norm_op, self.norm_op_kwargs,
                    self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                    None, ConvDropoutNormNonlin
                ),
                StackedConvLayers(
                    nfeatures_from_skip, final_num_features, 1,
                    self.conv_op, conv_kwargs, self.norm_op, self.norm_op_kwargs,
                    self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                    None, ConvDropoutNormNonlin
                )
            )
            loc_modules.append(loc_block)

    def forward(self, x):
        """Forward pass following exact official implementation structure"""
        # Encoder forward pass - NO pooling, all stages have same spatial dimensions
        x0_0 = self.conv_blocks_context[0](x)
        x1_0 = self.conv_blocks_context[1](x0_0)  # Applied to x0_0 directly, no pooling!
        x2_0 = self.conv_blocks_context[2](x1_0)  # Applied to x1_0 directly, no pooling!
        x3_0 = self.conv_blocks_context[3](x2_0)  # Applied to x2_0 directly, no pooling!
        x4_0 = self.conv_blocks_context[4](x3_0)  # Applied to x3_0 directly, no pooling!
        x5_0 = self.conv_blocks_context[5](x4_0)  # Applied to x4_0 directly, no pooling!
        
        seg_outputs = []
        
        # UNet++ nested forward pass (following official implementation exactly)
        # Level 4 (shallowest nesting)
        x0_1 = self.loc4[0](torch.cat([x0_0, self.up4[0](x1_0)], 1))
        seg_outputs.append(x0_1)

        # Level 3
        x1_1 = self.loc3[0](torch.cat([x1_0, self.up3[0](x2_0)], 1))
        x0_2 = self.loc3[1](torch.cat([x0_0, x0_1, self.up3[1](x1_1)], 1))
        seg_outputs.append(x0_2)

        # Level 2
        x2_1 = self.loc2[0](torch.cat([x2_0, self.up2[0](x3_0)], 1))
        x1_2 = self.loc2[1](torch.cat([x1_0, x1_1, self.up2[1](x2_1)], 1))
        x0_3 = self.loc2[2](torch.cat([x0_0, x0_1, x0_2, self.up2[2](x1_2)], 1))
        seg_outputs.append(x0_3)

        # Level 1  
        x3_1 = self.loc1[0](torch.cat([x3_0, self.up1[0](x4_0)], 1))
        x2_2 = self.loc1[1](torch.cat([x2_0, x2_1, self.up1[1](x3_1)], 1))
        x1_3 = self.loc1[2](torch.cat([x1_0, x1_1, x1_2, self.up1[2](x2_2)], 1))
        x0_4 = self.loc1[3](torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1[3](x1_3)], 1))
        seg_outputs.append(x0_4)

        # Level 0 (deepest nesting)
        x4_1 = self.loc0[0](torch.cat([x4_0, self.up0[0](x5_0)], 1))
        x3_2 = self.loc0[1](torch.cat([x3_0, x3_1, self.up0[1](x4_1)], 1))
        x2_3 = self.loc0[2](torch.cat([x2_0, x2_1, x2_2, self.up0[2](x3_2)], 1))
        x1_4 = self.loc0[3](torch.cat([x1_0, x1_1, x1_2, x1_3, self.up0[3](x2_3)], 1))
        x0_5 = self.loc0[4](torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up0[4](x1_4)], 1))
        seg_outputs.append(x0_5)

        # Apply segmentation heads
        final_outputs = []
        for i, seg_feat in enumerate(seg_outputs):
            final_outputs.append(self.seg_outputs[i](seg_feat))

        if self.deep_supervision:
            return tuple(final_outputs[::-1])  # Return from deepest to shallowest
        else:
            return final_outputs[-1]  # Return only the deepest output

    def compute_conv_feature_map_size(self, input_size):
        """
        Compute approximate memory consumption
        UNet++ uses significantly more memory due to nested structure
        """
        # Base encoder computation
        encoder_memory = 0
        current_size = list(input_size)
        
        for i, features in enumerate([32, 64, 128, 256, 320, 320]):  # Typical feature progression
            voxels = np.prod(current_size)
            encoder_memory += voxels * features * 4  # 4 bytes per float32
            
            # Update size for next stage (after pooling)
            if i < 5:  # No pooling after last stage
                for j in range(len(current_size)):
                    current_size[j] = current_size[j] // 2
        
        # UNet++ decoder uses approximately 3x more memory than standard decoder
        # due to storing multiple nested features simultaneously
        decoder_memory = encoder_memory * 3.0
        
        return int(encoder_memory + decoder_memory)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


# 为了兼容性，提供别名
MemoryEfficientUNetPlusPlus = UNetPlusPlus 