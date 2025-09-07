#!/usr/bin/env python3
"""
测试脚本：验证DualDecoderUNet训练器是否正常工作
"""

import torch
import numpy as np
from torch import nn

def test_dual_decoder_trainer():
    """
    测试DualDecoderUNet训练器的基本功能
    """
    print("开始测试DualDecoderUNet训练器...")
    
    try:
        # 导入训练器
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerDualDecoder import nnUNetTrainerDualDecoder
        print("✅ 成功导入nnUNetTrainerDualDecoder")
        
        # 测试网络构建
        arch_init_kwargs = {
            'conv_op': torch.nn.Conv3d,
            'norm_op': torch.nn.BatchNorm3d,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None,
            'dropout_op_kwargs': None,
            'nonlin': torch.nn.ReLU,
            'nonlin_kwargs': {'inplace': True},
            'n_stages': 5,
            'features_per_stage': [32, 64, 128, 256, 512],
            'kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'n_conv_per_stage': [2, 2, 2, 2, 2],
            'n_conv_per_stage_decoder': [2, 2, 2, 2],
            'conv_bias': False,
            'deep_supervision': False
        }
        
        network = nnUNetTrainerDualDecoder.build_network_architecture(
            architecture_class_name='dynamic_network_architectures.architectures.unet.DualDecoderUNet',
            arch_init_kwargs=arch_init_kwargs,
            arch_init_kwargs_req_import=[],
            num_input_channels=4,
            num_output_channels=2,
            enable_deep_supervision=False
        )
        print("✅ 成功构建DualDecoderUNet网络")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = network.to(device)
        
        # 创建测试数据
        batch_size = 1
        input_channels = 4
        patch_size = [64, 64, 64]
        
        test_input = torch.randn(batch_size, input_channels, *patch_size).to(device)
        
        with torch.no_grad():
            output = network(test_input)
            
        if isinstance(output, (tuple, list)):
            seg_output, reg_output = output
            print(f"✅ 前向传播成功")
            print(f"   分割输出形状: {seg_output.shape}")
            print(f"   回归输出形状: {reg_output.shape}")
        elif isinstance(output, dict):
            seg_output = output['segmentation']
            reg_output = output.get('regression', None)
            print(f"✅ 前向传播成功")
            print(f"   分割输出形状: {seg_output.shape}")
            if reg_output is not None:
                print(f"   回归输出形状: {reg_output.shape}")
        else:
            print(f"✅ 前向传播成功，输出形状: {output.shape}")
        
        # 测试损失计算
        print("\n测试损失计算...")
        
        # 创建模拟的训练批次
        batch = {
            'data': test_input,
            'target': torch.randint(0, 2, (batch_size, *patch_size)).to(device),
            'regression_target': torch.randn(batch_size, 1).to(device)
        }
        
        # 创建模拟的训练器实例（仅用于测试损失计算逻辑）
        class MockTrainer:
            def __init__(self):
                self.device = device
                self.network = network
                self.loss = nn.CrossEntropyLoss()
                self.optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
                self.grad_scaler = None
        
        mock_trainer = MockTrainer()
        
        # 模拟训练步骤的损失计算部分
        data = batch['data']
        target = batch['target']
        
        with torch.no_grad():
            output = network(data)
            
            if isinstance(output, (tuple, list)) and len(output) == 2:
                seg_output, reg_output = output
            elif isinstance(output, dict):
                seg_output = output['segmentation']
                reg_output = output.get('regression', None)
            else:
                seg_output = output
                reg_output = None
            
            # 计算分割损失（简化版）
            if seg_output.shape[1] > 1:  # 多类分割
                seg_loss = nn.CrossEntropyLoss()(seg_output, target.long())
            else:  # 二分类
                seg_loss = nn.BCEWithLogitsLoss()(seg_output, target.float().unsqueeze(1))
            
            total_loss = seg_loss
            
            # 如果有回归输出，计算回归损失
            if reg_output is not None and 'regression_target' in batch:
                reg_target = batch['regression_target']
                reg_loss = nn.MSELoss()(reg_output, reg_target)
                reg_weight = getattr(network, 'regression_loss_weight', 0.1)
                total_loss = seg_loss + reg_weight * reg_loss
                print(f"   分割损失: {seg_loss.item():.4f}")
                print(f"   回归损失: {reg_loss.item():.4f}")
                print(f"   回归权重: {reg_weight}")
                print(f"   总损失: {total_loss.item():.4f}")
            else:
                print(f"   分割损失: {seg_loss.item():.4f}")
                print(f"   总损失: {total_loss.item():.4f}")
        
        print("✅ 损失计算测试成功")
        
        # 测试不同epoch数的训练器
        print("\n测试不同epoch数的训练器...")
        
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerDualDecoder import (
            nnUNetTrainerDualDecoder_5epochs,
            nnUNetTrainerDualDecoder_10epochs,
            nnUNetTrainerDualDecoder_50epochs
        )
        
        epoch_trainers = {
            '5epochs': nnUNetTrainerDualDecoder_5epochs,
            '10epochs': nnUNetTrainerDualDecoder_10epochs,
            '50epochs': nnUNetTrainerDualDecoder_50epochs
        }
        
        for name, trainer_class in epoch_trainers.items():
            print(f"✅ 成功导入 {name} 训练器")
        
        print("\n🎉 所有测试通过！DualDecoderUNet训练器可以正常使用。")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_only():
    """
    仅测试网络本身，不依赖训练器
    """
    print("\n开始测试DualDecoderUNet网络本身...")
    
    try:
        # 直接导入网络
        from dynamic_network_architectures.architectures.unet import DualDecoderUNet
        print("✅ 成功导入DualDecoderUNet")
        
        # 创建网络实例
        network = DualDecoderUNet(
            input_channels=4,
            n_stages=5,
            features_per_stage=[32, 64, 128, 256, 512],
            conv_op=torch.nn.Conv3d,
            kernel_sizes=3,
            strides=[1, 2, 2, 2, 2],
            n_conv_per_stage=2,
            num_classes=2,
            num_regression_outputs=1,
            n_conv_per_stage_decoder=2,
            deep_supervision=False,
        )
        print("✅ 成功创建DualDecoderUNet实例")
        
        # 测试前向传播
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = network.to(device)
        
        test_input = torch.randn(1, 4, 64, 64, 64).to(device)
        
        with torch.no_grad():
            output = network(test_input)
            
        if isinstance(output, (tuple, list)):
            seg_output, reg_output = output
            print(f"✅ 网络前向传播成功")
            print(f"   分割输出形状: {seg_output.shape}")
            print(f"   回归输出形状: {reg_output.shape}")
            print(f"   回归损失权重: {network.regression_loss_weight}")
        
        return True
        
    except Exception as e:
        print(f"❌ 网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("DualDecoderUNet 训练器测试")
    print("=" * 50)
    
    # 首先测试网络本身
    network_ok = test_network_only()
    
    if network_ok:
        # 然后测试训练器
        trainer_ok = test_dual_decoder_trainer()
        
        if trainer_ok:
            print("\n🎉 所有测试通过！您可以开始使用DualDecoderUNet进行训练了。")
            print("\n下一步:")
            print("1. 准备您的数据集")
            print("2. 运行数据预处理")
            print("3. 使用以下命令开始训练:")
            print("   python examples/train_dual_decoder_example.py --dataset YourDataset")
        else:
            print("\n❌ 训练器测试失败，请检查错误信息")
    else:
        print("\n❌ 网络测试失败，请检查错误信息")