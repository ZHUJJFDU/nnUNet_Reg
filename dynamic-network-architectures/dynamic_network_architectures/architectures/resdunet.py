from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.abstract_arch import (
    AbstractDynamicNetworkArchitectures,
    test_submodules_loadable,
)
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class DenseBlock(nn.Module):
    """
    密集块 - DenseNet中的基本构建块
    每一层都与前面所有层连接
    """
    def __init__(self, in_channels, growth_rate, num_layers, conv_op, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, dropout_op=None, dropout_op_kwargs=None):
        super().__init__()
        
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        
        # 构建密集层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = self._make_layer(
                layer_in_channels, growth_rate, conv_op, norm_op, norm_op_kwargs,
                nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
            )
            self.layers.append(layer)
    
    def _make_layer(self, in_channels, out_channels, conv_op, norm_op, norm_op_kwargs,
                   nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs):
        """创建单个密集层"""
        layers = []
        
        # BN-ReLU-Conv 结构
        if norm_op is not None:
            layers.append(norm_op(in_channels, **norm_op_kwargs) if norm_op_kwargs else norm_op(in_channels))
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs) if nonlin_kwargs else nonlin())
        
        # 1x1 卷积降维（bottleneck）
        layers.append(conv_op(in_channels, 4 * out_channels, kernel_size=1))
        
        if norm_op is not None:
            layers.append(norm_op(4 * out_channels, **norm_op_kwargs) if norm_op_kwargs else norm_op(4 * out_channels))
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs) if nonlin_kwargs else nonlin())
        
        if dropout_op is not None:
            layers.append(dropout_op(**dropout_op_kwargs) if dropout_op_kwargs else dropout_op())
        
        # 3x3 卷积
        layers.append(conv_op(4 * out_channels, out_channels, kernel_size=3, padding=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        features = [x]
        
        for layer in self.layers:
            # 拼接所有之前的特征
            concat_features = torch.cat(features, dim=1)
            # 通过当前层
            new_feature = layer(concat_features)
            # 添加到特征列表
            features.append(new_feature)
        
        # 返回拼接的所有特征
        return torch.cat(features, dim=1)


class ResidualDenseBlock(nn.Module):
    """
    残差密集块 - 结合残差连接和密集连接
    """
    def __init__(self, in_channels, growth_rate, num_layers, conv_op, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, dropout_op=None, dropout_op_kwargs=None, 
                 residual_scale=0.2):
        super().__init__()
        
        self.residual_scale = residual_scale
        
        # 密集块
        self.dense_block = DenseBlock(
            in_channels, growth_rate, num_layers, conv_op, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
        )
        
        # 输出通道数
        out_channels = in_channels + num_layers * growth_rate
        
        # 1x1 卷积调整通道数
        self.conv_1x1 = conv_op(out_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        """前向传播"""
        identity = x
        
        # 通过密集块
        out = self.dense_block(x)
        
        # 1x1 卷积调整通道数
        out = self.conv_1x1(out)
        
        # 残差连接
        out = identity + out * self.residual_scale
        
        return out


class TransitionDown(nn.Module):
    """
    下采样过渡层
    """
    def __init__(self, in_channels, out_channels, conv_op, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, dropout_op=None, dropout_op_kwargs=None):
        super().__init__()
        
        layers = []
        
        # BN-ReLU
        if norm_op is not None:
            layers.append(norm_op(in_channels, **norm_op_kwargs) if norm_op_kwargs else norm_op(in_channels))
        if nonlin is not None:
            layers.append(nonlin(**nonlin_kwargs) if nonlin_kwargs else nonlin())
        
        # 1x1 卷积
        layers.append(conv_op(in_channels, out_channels, kernel_size=1))
        
        if dropout_op is not None:
            layers.append(dropout_op(**dropout_op_kwargs) if dropout_op_kwargs else dropout_op())
        
        # 2x2 平均池化
        if conv_op == nn.Conv3d:
            layers.append(nn.AvgPool3d(kernel_size=2, stride=2))
        else:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        self.transition = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.transition(x)


class TransitionUp(nn.Module):
    """
    上采样过渡层
    """
    def __init__(self, in_channels, out_channels, conv_op):
        super().__init__()
        
        # 转置卷积上采样
        if conv_op == nn.Conv3d:
            self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.up_conv(x)


class ResDUNetEncoder(nn.Module):
    """
    ResD-UNet编码器
    """
    def __init__(self, input_channels, n_stages, features_per_stage, conv_op, 
                 growth_rate=32, num_layers_per_block=4, norm_op=None, norm_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None, dropout_op=None, dropout_op_kwargs=None):
        super().__init__()
        
        self.n_stages = n_stages
        self.conv_op = conv_op
        self.output_channels = features_per_stage
        
        # 存储参数以供解码器使用
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        
        # 初始卷积
        self.init_conv = conv_op(input_channels, features_per_stage[0], kernel_size=3, padding=1)
        
        # 编码器阶段
        self.encoder_blocks = nn.ModuleList()
        self.transition_downs = nn.ModuleList()
        
        for i in range(n_stages):
            # 残差密集块
            block = ResidualDenseBlock(
                features_per_stage[i], growth_rate, num_layers_per_block, conv_op,
                norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
            )
            self.encoder_blocks.append(block)
            
            # 下采样（除了最后一个阶段）
            if i < n_stages - 1:
                transition = TransitionDown(
                    features_per_stage[i], features_per_stage[i + 1], conv_op,
                    norm_op, norm_op_kwargs, nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
                )
                self.transition_downs.append(transition)
    
    def forward(self, x):
        """前向传播"""
        # 初始卷积
        x = self.init_conv(x)
        
        # 存储跳跃连接
        skip_connections = []
        
        # 编码器阶段
        for i in range(self.n_stages):
            # 残差密集块
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            
            # 下采样
            if i < self.n_stages - 1:
                x = self.transition_downs[i](x)
        
        return skip_connections


class ResDUNetDecoder(nn.Module):
    """
    ResD-UNet解码器
    """
    def __init__(self, encoder, num_classes, growth_rate=32, num_layers_per_block=4, 
                 deep_supervision=False):
        super().__init__()
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        
        features = encoder.output_channels
        self.n_stages = len(features)
        
        # 解码器阶段
        self.transition_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.n_stages - 1, 0, -1):
            # 上采样
            up_conv = TransitionUp(features[i], features[i - 1], encoder.conv_op)
            self.transition_ups.append(up_conv)
            
            # 拼接后的通道数
            concat_channels = features[i - 1] * 2
            
            # 残差密集块
            block = ResidualDenseBlock(
                concat_channels, growth_rate, num_layers_per_block, encoder.conv_op,
                encoder.norm_op, encoder.norm_op_kwargs, encoder.nonlin, encoder.nonlin_kwargs,
                encoder.dropout_op, encoder.dropout_op_kwargs
            )
            self.decoder_blocks.append(block)
        
        # 最终分类层
        if deep_supervision:
            self.final_convs = nn.ModuleList([
                encoder.conv_op(features[i], num_classes, kernel_size=1)
                for i in range(self.n_stages - 1)
            ])
        else:
            self.final_convs = nn.ModuleList([
                encoder.conv_op(features[0], num_classes, kernel_size=1)
            ])
    
    def forward(self, skip_connections):
        """前向传播"""
        # 从最深层开始
        x = skip_connections[-1]
        outputs = []
        
        # 解码器阶段
        for i, (up_conv, decoder_block) in enumerate(zip(self.transition_ups, self.decoder_blocks)):
            # 上采样
            x = up_conv(x)
            
            # 获取对应的跳跃连接
            skip_idx = self.n_stages - 2 - i
            skip = skip_connections[skip_idx]
            
            # 拼接特征
            x = torch.cat([x, skip], dim=1)
            
            # 残差密集块
            x = decoder_block(x)
            
            # 深度监督输出
            if self.deep_supervision and i < len(self.final_convs) - 1:
                output = self.final_convs[i](x)
                outputs.append(output)
        
        # 最终输出
        final_output = self.final_convs[-1](x)
        outputs.append(final_output)
        
        if self.deep_supervision:
            return outputs[::-1]  # 从浅到深的顺序
        else:
            return final_output
    
    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        # 简化实现
        return sum([
            torch.prod(torch.tensor(input_size), dtype=torch.float) * f 
            for f in self.encoder.output_channels
        ]) * 1.5  # 密集连接增加了计算量


class ResDUNet(AbstractDynamicNetworkArchitectures):
    """
    ResD-UNet 实现
    结合了残差连接和密集连接的U-Net变体
    基于DenseNet和ResNet的思想
    """
    
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        num_classes: int,
        growth_rate: int = 32,
        num_layers_per_block: int = 4,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
    ):
        """
        初始化ResD-UNet
        
        Args:
            growth_rate: 密集块的增长率
            num_layers_per_block: 每个密集块的层数
        """
        super().__init__()

        # 设置键值
        self.key_to_encoder = "encoder"
        self.key_to_stem = "encoder.init_conv"
        self.keys_to_in_proj = ("encoder.init_conv",)

        # 参数处理
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        
        assert len(features_per_stage) == n_stages, (
            f"features_per_stage must have as many entries as we have resolution stages. "
            f"here: {n_stages}. features_per_stage: {features_per_stage}"
        )

        # 创建编码器
        self.encoder = ResDUNetEncoder(
            input_channels, n_stages, features_per_stage, conv_op,
            growth_rate, num_layers_per_block, norm_op, norm_op_kwargs,
            nonlin, nonlin_kwargs, dropout_op, dropout_op_kwargs
        )

        # 创建解码器
        self.decoder = ResDUNetDecoder(
            self.encoder, num_classes, growth_rate, num_layers_per_block, deep_supervision
        )

    def forward(self, x):
        """前向传播"""
        skip_connections = self.encoder(x)
        return self.decoder(skip_connections)

    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        encoder_size = sum([
            torch.prod(torch.tensor(input_size), dtype=torch.float) * f 
            for f in self.encoder.output_channels
        ])
        decoder_size = self.decoder.compute_conv_feature_map_size(input_size)
        return encoder_size + decoder_size

    @staticmethod
    def initialize(module):
        """权重初始化"""
        def _init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        module.apply(_init_weights)


# 别名
ResidualDenseUNet = ResDUNet 