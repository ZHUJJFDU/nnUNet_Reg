from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
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


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    压缩空间维度，关注通道间关系
    """
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        # 根据维度动态选择自适应池化操作
        self._avg_pool = nn.AdaptiveAvgPool3d(1)  # 默认为3D
        self._max_pool = nn.AdaptiveMaxPool3d(1)
        
        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self._avg_pool(x))
        max_out = self.fc(self._max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
    def set_conv_dim(self, dim):
        """根据输入维度设置池化和卷积操作"""
        if dim == 2:
            self._avg_pool = nn.AdaptiveAvgPool2d(1)
            self._max_pool = nn.AdaptiveMaxPool2d(1)
            # 更新FC层中的卷积维度
            new_fc = nn.Sequential(
                nn.Conv2d(self.fc[0].in_channels, self.fc[0].out_channels, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.fc[2].in_channels, self.fc[2].out_channels, 1, bias=False)
            )
            # 复制权重
            new_fc[0].weight.data = self.fc[0].weight.data.squeeze(-1)
            new_fc[2].weight.data = self.fc[2].weight.data.squeeze(-1)
            self.fc = new_fc
        elif dim == 3:
            self._avg_pool = nn.AdaptiveAvgPool3d(1)
            self._max_pool = nn.AdaptiveMaxPool3d(1)
            # FC层已经是3D的，不需更新


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    聚焦于空间位置的重要性
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "空间注意力卷积核大小必须是3或7"
        padding = 3 if kernel_size == 7 else 1
        
        # 默认为3D卷积
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接特征
        out = torch.cat([avg_out, max_out], dim=1)
        
        # 应用卷积和sigmoid
        out = self.conv(out)
        return self.sigmoid(out)
    
    def set_conv_dim(self, dim):
        """根据输入维度设置卷积操作"""
        kernel_size = self.conv.kernel_size[0]
        padding = self.conv.padding[0]
        
        if dim == 2:
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        elif dim == 3:
            self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)


class CBAM(nn.Module):
    """
    CBAM注意力模块
    结合通道注意力和空间注意力
    """
    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel_size)
    
    def forward(self, x):
        # print("cbam")
        # 先应用通道注意力
        x = x * self.ca(x)
        # 再应用空间注意力
        x = x * self.sa(x)
        return x
    
    def set_conv_dim(self, dim):
        """设置所有子模块的卷积维度"""
        self.ca.set_conv_dim(dim)
        self.sa.set_conv_dim(dim)


class CBAMPlainConvUNet(AbstractDynamicNetworkArchitectures):
    """
    带有CBAM注意力模块的PlainConvUNet
    通过在编码器的每个阶段添加CBAM模块增强特征提取能力
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
        cbam_spatial_kernel_size: int = 7
    ):
        """
        初始化CBAMPlainConvUNet
        
        参数与PlainConvUNet相同，增加了两个额外参数:
        cbam_reduction_ratio: 通道注意力中的通道降维比例
        cbam_spatial_kernel_size: 空间注意力的卷积核大小
        """
        super().__init__()
        
        # 保持与PlainConvUNet相同的键
        self.key_to_encoder = "encoder.stages"  # Contains the stem as well.
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",  # duplicate of above
        )

        # 参数处理
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage必须具有与分辨率阶段数相同的条目。"
            f"这里: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder必须比分辨率阶段数少一个条目。"
            f"这里: {n_stages} 个阶段, 应该有 {n_stages - 1} 个条目。"
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
        
        # 创建解码器
        self.decoder = UNetDecoder(
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first
        )
        
        # 创建CBAM模块
        self.cbam_modules = nn.ModuleList()
        
        # 获取卷积维度
        conv_dim = convert_conv_op_to_dim(conv_op)
        
        # 为每个阶段创建CBAM模块
        if isinstance(features_per_stage, (list, tuple)):
            for feature_num in features_per_stage:
                cbam = CBAM(feature_num, cbam_reduction_ratio, cbam_spatial_kernel_size)
                cbam.set_conv_dim(conv_dim)  # 设置正确的卷积维度
                self.cbam_modules.append(cbam)
        else:
            # 如果features_per_stage是单个整数，每个阶段使用相同数量的特征
            for _ in range(n_stages):
                cbam = CBAM(features_per_stage, cbam_reduction_ratio, cbam_spatial_kernel_size)
                cbam.set_conv_dim(conv_dim)  # 设置正确的卷积维度
                self.cbam_modules.append(cbam)

    def forward(self, x):
        """
        前向传播函数
        首先通过编码器获取特征，然后应用CBAM，最后通过解码器获得输出
        """
        # 获取编码器输出（跳跃连接）
        skips = self.encoder(x)
        
        # 应用CBAM模块到跳跃连接
        enhanced_skips = []
        for i, skip in enumerate(skips):
            if i < len(self.cbam_modules):
                # 应用CBAM增强特征
                enhanced_skip = self.cbam_modules[i](skip)
                enhanced_skips.append(enhanced_skip)
            else:
                # 如果没有对应的CBAM模块，保持原样
                enhanced_skips.append(skip)
        
        # 使用增强的跳跃连接进行解码
        return self.decoder(enhanced_skips)

    def compute_conv_feature_map_size(self, input_size):
        """
        计算卷积特征图大小
        与PlainConvUNet相同的实现
        """
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "请只提供不包含颜色/特征通道或批次通道的图像大小。"
            "不要提供 input_size=(b, c, x, y(, z))，"
            "而是提供 input_size=(x, y(, z))！"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        """
        初始化网络权重
        与PlainConvUNet相同的实现
        """
        InitWeights_He(1e-2)(module)


if __name__ == "__main__":
    # 测试代码
    data = torch.rand((1, 4, 128, 128, 128))

    # 创建带有CBAM的UNet
    model = CBAMPlainConvUNet(
        4,
        6,
        (32, 64, 125, 256, 320, 320),
        nn.Conv3d,
        3,
        (1, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
        4,
        (2, 2, 2, 2, 2),
        False,
        nn.BatchNorm3d,
        None,
        None,
        None,
        nn.ReLU,
        deep_supervision=True,
    )

    # 测试模块是否可加载
    test_submodules_loadable(model)
    
    print(f"模型计算的特征图大小: {model.compute_conv_feature_map_size(data.shape[2:])}")
    
    # 测试2D模型
    data_2d = torch.rand((1, 4, 512, 512))

    model_2d = CBAMPlainConvUNet(
        4,
        8,
        (32, 64, 125, 256, 512, 512, 512, 512),
        nn.Conv2d,
        3,
        (1, 2, 2, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2, 2, 2),
        4,
        (2, 2, 2, 2, 2, 2, 2),
        False,
        nn.BatchNorm2d,
        None,
        None,
        None,
        nn.ReLU,
        deep_supervision=True,
    )
    
    test_submodules_loadable(model_2d)
    print(f"2D模型计算的特征图大小: {model_2d.compute_conv_feature_map_size(data_2d.shape[2:])}") 