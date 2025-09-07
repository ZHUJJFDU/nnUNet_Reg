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
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


class AttentionGate(nn.Module):
    """
    注意力门控模块
    用于在跳跃连接中突出显示相关特征并抑制无关特征
    """
    def __init__(self, F_g, F_l, F_int, conv_op=nn.Conv3d):
        """
        Args:
            F_g: 门控信号的通道数 (来自解码器的上采样特征)
            F_l: 输入特征的通道数 (来自编码器的跳跃连接)
            F_int: 中间层的通道数
            conv_op: 卷积操作类型 (Conv2d 或 Conv3d)
        """
        super(AttentionGate, self).__init__()
        
        self.conv_op = conv_op
        
        # 门控信号的卷积层
        self.W_g = nn.Sequential(
            conv_op(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int) if conv_op == nn.Conv3d else nn.BatchNorm2d(F_int)
        )
        
        # 输入特征的卷积层
        self.W_x = nn.Sequential(
            conv_op(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int) if conv_op == nn.Conv3d else nn.BatchNorm2d(F_int)
        )
        
        # 注意力系数生成
        self.psi = nn.Sequential(
            conv_op(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1) if conv_op == nn.Conv3d else nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """
        Args:
            g: 门控信号 (来自解码器的上采样特征)
            x: 输入特征 (来自编码器的跳跃连接)
        Returns:
            注意力加权后的特征
        """
        # 获取输入特征的空间尺寸
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 如果尺寸不匹配，对门控信号进行上采样
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear' if len(x1.shape) == 5 else 'bilinear', align_corners=False)
        
        # 计算注意力系数
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 应用注意力权重
        return x * psi


class AttUNetDecoder(nn.Module):
    """
    带有注意力门控的U-Net解码器
    """
    def __init__(self, encoder, num_classes, n_conv_per_stage_decoder, deep_supervision=False, nonlin_first=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder
        
        # 获取编码器的特征通道数
        encoder_features = encoder.output_channels
        
        # 构建解码器阶段
        stages = []
        attention_gates = []
        
        # 从最深层开始构建解码器
        for i in range(len(encoder_features) - 2, -1, -1):
            # 当前层和上一层的特征通道数
            input_features = encoder_features[i + 1]
            skip_features = encoder_features[i]
            output_features = encoder_features[i]
            
            # 注意力门控 - 使用正确的通道数
            attention_gate = AttentionGate(
                F_g=output_features,  # 上采样后的特征通道数
                F_l=skip_features,    # 跳跃连接的特征通道数
                F_int=skip_features // 2,
                conv_op=encoder.conv_op
            )
            attention_gates.append(attention_gate)
            
            # 上采样 + 卷积
            stage = nn.ModuleList([
                # 上采样
                nn.ConvTranspose3d(input_features, output_features, kernel_size=2, stride=2) 
                if encoder.conv_op == nn.Conv3d else 
                nn.ConvTranspose2d(input_features, output_features, kernel_size=2, stride=2),
                
                # 卷积块
                self._make_conv_block(
                    output_features + skip_features,  # 拼接后的通道数
                    output_features,
                    n_conv_per_stage_decoder[len(encoder_features) - 2 - i],
                    encoder.conv_op,
                    encoder.norm_op,
                    encoder.norm_op_kwargs,
                    encoder.dropout_op,
                    encoder.dropout_op_kwargs,
                    encoder.nonlin,
                    encoder.nonlin_kwargs,
                    nonlin_first
                )
            ])
            stages.append(stage)
        
        self.stages = nn.ModuleList(stages)
        self.attention_gates = nn.ModuleList(attention_gates)
        
        # 我们总是构建深度监督输出，这样我们就可以总是加载参数。如果我们不这样做，
        # 那么用deep_supervision=True训练的模型在推理时就不能轻易加载，
        # 因为推理时deep_supervision=False。这只是为了方便。
        # 复制标准UNetDecoder的逻辑
        seg_outputs = []
        n_stages_encoder = len(encoder_features)
        for s in range(1, n_stages_encoder):
            input_features_skip = encoder_features[-(s + 1)]
            seg_outputs.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))
        
        self.seg_outputs = nn.ModuleList(seg_outputs)
    
    def _make_conv_block(self, input_channels, output_channels, n_convs, conv_op, norm_op, norm_op_kwargs, 
                        dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first):
        """创建卷积块"""
        layers = []
        
        for i in range(n_convs):
            in_ch = input_channels if i == 0 else output_channels
            
            if nonlin_first:
                if i > 0:  # 第一层不加激活
                    layers.append(nonlin(**nonlin_kwargs) if nonlin_kwargs else nonlin())
                if norm_op is not None:
                    layers.append(norm_op(in_ch, **norm_op_kwargs) if norm_op_kwargs else norm_op(in_ch))
                layers.append(conv_op(in_ch, output_channels, kernel_size=3, padding=1))
            else:
                layers.append(conv_op(in_ch, output_channels, kernel_size=3, padding=1))
                if norm_op is not None:
                    layers.append(norm_op(output_channels, **norm_op_kwargs) if norm_op_kwargs else norm_op(output_channels))
                if i < n_convs - 1:  # 最后一层不加激活
                    layers.append(nonlin(**nonlin_kwargs) if nonlin_kwargs else nonlin())
            
            if dropout_op is not None and i < n_convs - 1:
                layers.append(dropout_op(**dropout_op_kwargs) if dropout_op_kwargs else dropout_op())
        
        return nn.Sequential(*layers)
    
    def forward(self, skips):
        """
        Args:
            skips: 来自编码器的跳跃连接特征列表
        """
        # 从最深层开始解码
        x = skips[-1]  # 最深层特征
        seg_outputs = []
        
        for i, (stage, attention_gate) in enumerate(zip(self.stages, self.attention_gates)):
            # 上采样
            x = stage[0](x)
            
            # 获取对应的跳跃连接
            skip_idx = len(skips) - 2 - i
            skip = skips[skip_idx]
            
            # 确保上采样后的特征与跳跃连接特征的空间尺寸匹配
            if x.shape[2:] != skip.shape[2:]:
                # 如果尺寸不匹配，调整上采样特征的尺寸
                if len(x.shape) == 5:  # 3D
                    x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
                else:  # 2D
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # 应用注意力门控
            skip_att = attention_gate(x, skip)
            
            # 拼接特征
            x = torch.cat([x, skip_att], dim=1)
            
            # 卷积处理
            x = stage[1](x)
            
            # 深度监督输出 - 复制标准UNetDecoder的逻辑
            if self.deep_supervision:
                seg_outputs.append(self.seg_outputs[i](x))  # type: ignore
            elif i == (len(self.stages) - 1):
                seg_outputs.append(self.seg_outputs[-1](x))  # type: ignore
        
        # 反转seg输出，使最大的分割预测首先返回 - 复制标准UNetDecoder逻辑
        seg_outputs = seg_outputs[::-1]
        
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
    
    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        # 简化实现，返回与编码器相同的大小
        return sum([
            torch.prod(torch.tensor(input_size), dtype=torch.float) * f 
            for f in self.encoder.output_channels
        ])


class PlainAttUNet(AbstractDynamicNetworkArchitectures):
    """
    Attention U-Net 实现
    基于论文: "Attention U-Net: Learning Where to Look for the Pancreas"
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
    ):
        """
        初始化Attention U-Net
        
        参数与PlainConvUNet相同，但使用了带注意力门控的解码器
        """
        super().__init__()

        # 设置键值，与PlainConvUNet保持一致
        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",
        )

        # 参数验证和处理
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
            
        assert len(n_conv_per_stage) == n_stages, (
            f"n_conv_per_stage must have as many entries as we have resolution stages. "
            f"here: {n_stages}. n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            f"n_conv_per_stage_decoder must have one less entries as we have resolution stages. "
            f"here: {n_stages} stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )

        # 创建编码器
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )

        # 创建带注意力门控的解码器
        self.decoder = AttUNetDecoder(
            self.encoder, 
            num_classes, 
            n_conv_per_stage_decoder, 
            deep_supervision, 
            nonlin_first=nonlin_first
        )

    def forward(self, x):
        """前向传播"""
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        """计算卷积特征图大小"""
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + \
               self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        """权重初始化"""
        InitWeights_He(1e-2)(module)


# 为了兼容性，提供一个别名
PlainAttentionUNet = PlainAttUNet
