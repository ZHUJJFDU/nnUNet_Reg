import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这行导入
from typing import Union, List, Tuple, Callable
import numpy as np

from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, attention_dropout_rate=0.0):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout_rate=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, hidden_size, in_channels=1, dropout_rate=0.0):
        super(Embeddings, self).__init__()
        self.patch_embeddings = nn.Conv3d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=1,
                                       stride=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, d, h, w)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # Dynamically create position embeddings
        n_patches = x.shape[1]
        position_embeddings = nn.Parameter(torch.zeros(1, n_patches, x.shape[-1]), requires_grad=True).to(x.device)

        embeddings = x + position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_dim, dropout_rate=0.0, attention_dropout_rate=0.0):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_size, num_heads, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, dropout_rate=0.0, attention_dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, mlp_dim, feat_channels=1, 
                 dropout_rate=0.0, attention_dropout_rate=0.0):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(hidden_size, in_channels=feat_channels, dropout_rate=dropout_rate)
        self.encoder = Encoder(hidden_size, num_layers, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)

        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, hidden, d, h, w)
        d, h, w = input_ids.shape[2:]
        x = encoded.permute(0, 2, 1)
        encoded = x.contiguous().view(B, hidden, d, h, w)
        return encoded


class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)
        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs['p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None):
        super(StackedConvLayers, self).__init__()
        
        if first_stride is not None:
            self.convs = nn.Sequential(
                ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, conv_op,
                                    {**conv_kwargs, 'stride': first_stride}, norm_op, norm_op_kwargs,
                                    dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs),
                *[ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, conv_op,
                                      conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                      nonlin, nonlin_kwargs) for _ in range(num_convs - 1)]
            )
        else:
            self.convs = nn.Sequential(
                ConvDropoutNormNonlin(input_feature_channels, output_feature_channels, conv_op,
                                    conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                    nonlin, nonlin_kwargs),
                *[ConvDropoutNormNonlin(output_feature_channels, output_feature_channels, conv_op,
                                      conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                      nonlin, nonlin_kwargs) for _ in range(num_convs - 1)]
            )

    def forward(self, x):
        return self.convs(x)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                       align_corners=self.align_corners)


class TransUNet(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size=(64, 64, 64), hidden_size=768, num_layers=12, num_heads=12, mlp_dim=3072,
                 n_stages=6, features_per_stage=(32, 64, 128, 256, 320, 320), conv_op=nn.Conv3d, kernel_sizes=3, strides=(1, 2, 2, 2, 2, 2),
                 n_conv_per_stage=2, n_conv_per_stage_decoder=(2, 2, 2, 2, 2), conv_bias=True, norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                 conv_kwargs=None, dropout_rate=0.0, attention_dropout_rate=0.0):
        super().__init__()
        
        if isinstance(conv_op, str):
            conv_op = locate(conv_op)
        if isinstance(norm_op, str):
            norm_op = locate(norm_op)
        if isinstance(dropout_op, str):
            dropout_op = locate(dropout_op)
        if isinstance(nonlin, str):
            nonlin = locate(nonlin)
            
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': conv_bias}
            
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
            
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.0, 'inplace': True}
            
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            
        self.conv_op = conv_op
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.deep_supervision = deep_supervision
        self.patch_size = patch_size
        self.strides = strides  # 保存strides信息
        
        if not isinstance(features_per_stage, (list, tuple)):
            features_per_stage = [features_per_stage] * n_stages
        if not isinstance(n_conv_per_stage, (list, tuple)):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if not isinstance(n_conv_per_stage_decoder, (list, tuple)):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        if not isinstance(kernel_sizes, (list, tuple)):
            kernel_sizes = [kernel_sizes] * n_stages
        if not isinstance(strides, (list, tuple)):
            strides = [strides] * n_stages
            
        # Build encoder
        stages = []
        stage_output_features = []
        
        # First stage
        stage_output_features.append(features_per_stage[0])
        stages.append(StackedConvLayers(
            input_channels, features_per_stage[0], n_conv_per_stage[0], conv_op, conv_kwargs,
            norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, strides[0]
        ))
        
        # Subsequent stages
        for s in range(1, n_stages):
            stage_output_features.append(features_per_stage[s])
            stages.append(StackedConvLayers(
                stage_output_features[-2], features_per_stage[s], n_conv_per_stage[s], conv_op, conv_kwargs,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, strides[s]
            ))
            
        self.stages = nn.ModuleList(stages)
        
        # Calculate feature map size for transformer
        # Use the patch_size to dynamically calculate the feature map size
        if isinstance(patch_size, int):
            patch_size = [patch_size] * len(strides[0]) if isinstance(strides[0], (list, tuple)) else [patch_size] * 3
        feat_size = [patch_size[i] // int(np.prod([s[i] if isinstance(s, (list, tuple)) else s for s in strides])) for i in range(len(patch_size))]
        
        # Transformer (replaces the bottleneck)
        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_layers=num_layers, 
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            feat_channels=features_per_stage[-1],
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate
        )
        
        # Projection layer to match encoder features
        self.transformer_proj = conv_op(hidden_size, features_per_stage[-1], kernel_size=1)
        
        # Build decoder
        decoder_stages = []
        
        for s in range(n_stages - 1):
            input_features = features_per_stage[-(s+1)] + features_per_stage[-(s+2)]
            decoder_stages.append(nn.Sequential(
                Upsample(scale_factor=strides[-(s+1)], mode='trilinear' if conv_op == nn.Conv3d else 'bilinear'),
                StackedConvLayers(
                    input_features, features_per_stage[-(s+2)], n_conv_per_stage_decoder[s], conv_op, conv_kwargs,
                    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
                )
            ))
            
        self.decoder_stages = nn.ModuleList(decoder_stages)
        
        # Add decoder attribute for nnUNet compatibility
        self.decoder = type('DummyDecoder', (), {
            'deep_supervision': deep_supervision
        })()
        
        # Final segmentation layers
        self.seg_layers = nn.ModuleList([
            conv_op(features_per_stage[0], num_classes, kernel_size=1)
        ])
        
        if deep_supervision:
            # Create deep supervision layers for decoder stages
            for s in range(n_stages - 1):
                self.seg_layers.append(
                    conv_op(features_per_stage[-(s+2)], num_classes, kernel_size=1)
                )
                
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x):
        skips = []
        input_shape = x.shape[2:]  # 获取输入的空间维度
        
        # Encoder
        for s in range(self.n_stages):
            x = self.stages[s](x)
            if s < self.n_stages - 1:  # Don't store the last stage as skip
                skips.append(x)
                
        # Transformer bottleneck
        x = self.transformer(x)
        x = self.transformer_proj(x)
        
        # Decoder - 完全重写上采样逻辑
        outputs = []
        for s in range(len(self.decoder_stages)):
            # 获取对应的skip connection的尺寸作为目标尺寸
            target_size = skips[-(s+1)].shape[2:]
            
            # 直接上采样到skip connection的尺寸
            x = F.interpolate(
                x, 
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            
            # Skip connection
            x = torch.cat([x, skips[-(s+1)]], dim=1)
            
            # 只使用卷积层，跳过Upsample层
            x = self.decoder_stages[s][1](x)
            
            if self.deep_supervision:
                deep_sup_output = self.seg_layers[s+1](x)
                # 确保深度监督输出上采样到输入尺寸
                deep_sup_output = F.interpolate(
                    deep_sup_output,
                    size=input_shape,
                    mode='trilinear',
                    align_corners=False
                )
                outputs.append(deep_sup_output)
                
        # Final output - 确保最终输出上采样到输入尺寸
        seg_output = self.seg_layers[0](x)
        
        # 检查当前输出尺寸，如果不匹配则强制上采样
        current_size = seg_output.shape[2:]
        if current_size != input_shape:
            seg_output = F.interpolate(
                seg_output,
                size=input_shape,
                mode='trilinear',
                align_corners=False
            )
        
        if self.deep_supervision:
            outputs.append(seg_output)
            return outputs[::-1]  # Return in order from coarse to fine
        else:
            return seg_output


def locate(name):
    """Locate a class or function by its string name"""
    import importlib
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


if __name__ == "__main__":
    # Test the model
    model = TransUNet(
        input_channels=1,
        num_classes=3,
        patch_size=(64, 64, 64),
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        mlp_dim=3072,
        deep_supervision=True
    )
    
    x = torch.randn(2, 1, 64, 64, 64)
    output = model(x)
    print(f"Input shape: {x.shape}")
    if isinstance(output, list):
        for i, out in enumerate(output):
            print(f"Output {i} shape: {out.shape}")
    else:
        print(f"Output shape: {output.shape}")