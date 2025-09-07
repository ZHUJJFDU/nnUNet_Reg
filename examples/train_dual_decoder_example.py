#!/usr/bin/env python3
"""
示例脚本：展示如何训练DualDecoderUNet网络
这个脚本演示了如何使用新创建的双解码器网络进行训练
"""

import os
import argparse
from nnunetv2.run.run_training import run_training
import torch

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='训练DualDecoderUNet网络')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称或ID (例如: Dataset001_BrainTumour)')
    parser.add_argument('--configuration', type=str, default="3d_fullres", help='配置 (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)')
    parser.add_argument('--fold', type=int, default=0, help='交叉验证折数 (0-4, 或者 all)')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数，如设为None则使用默认值')
    parser.add_argument('--trainer', type=str, default=None, help='训练器名称，如果不指定则根据epochs自动选择')
    parser.add_argument('--plans', type=str, default="nnUNetPlans", help='计划文件标识符')
    parser.add_argument('--pretrained_weights', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--device', type=str, default=None, help='设备 (cuda, cpu, 或者具体的GPU ID如cuda:0)')
    parser.add_argument('--disable_amp', action='store_true', help='禁用自动混合精度训练')
    parser.add_argument('--continue_training', action='store_true', help='继续之前的训练')
    parser.add_argument('--only_run_validation', action='store_true', help='只运行验证')
    parser.add_argument('--disable_checkpointing', action='store_true', help='禁用检查点保存')
    
    args = parser.parse_args()
    
    # 设置环境变量以优化性能
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # 确定使用的训练器
    if args.trainer is None:
        if args.epochs is None:
            trainer_class_name = "nnUNetTrainerDualDecoder"  # 默认训练器
        else:
            # 根据epoch数选择对应的训练器
            if args.epochs == 5:
                trainer_class_name = "nnUNetTrainerDualDecoder_5epochs"
            elif args.epochs == 10:
                trainer_class_name = "nnUNetTrainerDualDecoder_10epochs"
            elif args.epochs == 20:
                trainer_class_name = "nnUNetTrainerDualDecoder_20epochs"
            elif args.epochs == 50:
                trainer_class_name = "nnUNetTrainerDualDecoder_50epochs"
            elif args.epochs == 100:
                trainer_class_name = "nnUNetTrainerDualDecoder_100epochs"
            else:
                trainer_class_name = "nnUNetTrainerDualDecoder"
                print(f"警告: 没有找到{args.epochs}轮的预设训练器，使用默认训练器")
    else:
        trainer_class_name = args.trainer
    
    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("DualDecoderUNet 训练配置")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"配置: {args.configuration}")
    print(f"折数: {args.fold}")
    print(f"训练器: {trainer_class_name}")
    print(f"计划文件: {args.plans}")
    print(f"设备: {device}")
    print(f"自动混合精度: {'禁用' if args.disable_amp else '启用'}")
    if args.pretrained_weights:
        print(f"预训练权重: {args.pretrained_weights}")
    print("=" * 60)
    
    # 运行训练
    try:
        run_training(
            dataset_name_or_id=args.dataset,
            configuration=args.configuration,
            fold=args.fold,
            trainer_class_name=trainer_class_name,
            plans_identifier=args.plans,
            pretrained_weights=args.pretrained_weights,
            num_gpus=1,
            use_compressed_data=False,
            export_validation_probabilities=True,
            continue_training=args.continue_training,
            only_run_validation=args.only_run_validation,
            disable_checkpointing=args.disable_checkpointing,
            device=device
        )
        print("\n训练完成！")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查数据集是否正确预处理")
        print("2. 确认数据集路径设置正确")
        print("3. 检查GPU内存是否足够")
        print("4. 尝试使用较小的batch size")
        print("5. 如果是回归任务，确保数据包含回归目标")
        raise

if __name__ == '__main__':
    main()