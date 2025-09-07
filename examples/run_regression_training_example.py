#!/usr/bin/env python

"""
示例脚本：展示如何使用run_regression_training进行回归训练
支持选择不同轮数的训练器
"""

import os
import argparse
from nnunetv2.run.run_regression_training import run_regression_training
import torch

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行nnUNet回归训练')
    parser.add_argument('--dataset', type=str, default="Dataset102_Reg", help='数据集名称')
    parser.add_argument('--configuration', type=str, default="3d_fullres", help='配置')
    parser.add_argument('--fold', type=int, default=0, help='交叉验证折数')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数，如设为None则使用默认值')
    parser.add_argument('--reg_weight', type=float, default=1.0, help='回归损失权重')
    parser.add_argument('--reg_loss', type=str, default="mse", choices=["mse", "mae"], help='回归损失类型')
    parser.add_argument('--reg_key', type=str, default="bulla_thickness", help='回归值键名')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--disable_amp', action='store_true', help='禁用自动混合精度(AMP)训练，解决数据类型不匹配问题')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # 基本参数
    dataset_name = args.dataset
    configuration = args.configuration
    fold = args.fold
    
    # 回归特定参数
    regression_weight = args.reg_weight
    regression_loss_type = args.reg_loss
    regression_key = args.reg_key
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 选择训练器类名
    if args.epochs is None:
        trainer_class_name = "RegnnUNetTrainer"  # 默认训练器
    else:
        # 检查是否有匹配的预设epoch数
        if args.epochs == 5:
            trainer_class_name = "RegnnUNetTrainer_5epochs"
        elif args.epochs == 10:
            trainer_class_name = "RegnnUNetTrainer_10epochs"
        elif args.epochs == 20:
            trainer_class_name = "RegnnUNetTrainer_20epochs"
        elif args.epochs == 50:
            trainer_class_name = "RegnnUNetTrainer_50epochs"
        elif args.epochs == 100:
            trainer_class_name = "RegnnUNetTrainer_100epochs"
        elif args.epochs == 250:
            trainer_class_name = "RegnnUNetTrainer_250epochs"
        elif args.epochs == 500:
            trainer_class_name = "RegnnUNetTrainer_500epochs"
        elif args.epochs == 1000:
            trainer_class_name = "RegnnUNetTrainer_1000epochs"
        else:
            # 如果没有匹配的预设，使用自定义epoch数
            trainer_class_name = "RegnnUNetTrainer_CustomEpochs"
    
    print(f"使用训练器: {trainer_class_name}")
    print(f"回归权重: {regression_weight}")
    print(f"回归损失类型: {regression_loss_type}")
    print(f"回归键: {regression_key}")
    print(f"调试模式: {'启用' if args.debug else '禁用'}")
    print(f"自动混合精度(AMP): {'禁用' if args.disable_amp else '启用'}")
    
    # 额外参数字典，用于CustomEpochs
    additional_kwargs = {}
    if trainer_class_name == "RegnnUNetTrainer_CustomEpochs":
        additional_kwargs['num_epochs'] = args.epochs
    
    # 创建自定义训练器设置类
    class CustomTrainerSetup:
        def __init__(self, trainer):
            self.trainer = trainer
            # 在这里可以添加更多自定义初始化逻辑
        
        def apply_settings(self):
            # 如果禁用AMP，在训练器初始化后设置
            if args.disable_amp:
                self.trainer.amp = False
                print("已禁用自动混合精度(AMP)训练")
    
    # 运行训练
    run_regression_training(
        dataset_name_or_id=dataset_name,
        configuration=configuration,
        fold=fold,
        trainer_class_name=trainer_class_name,
        plans_identifier="nnUNetPlans",
        regression_weight=regression_weight,
        regression_loss_type=regression_loss_type,
        regression_key=regression_key,
        device=device,
        debug=args.debug,
        disable_amp=args.disable_amp,  # 传递AMP禁用参数
        **additional_kwargs
    )

# 命令行使用示例:
# python examples/run_regression_training_example.py --dataset Dataset103_quan --configuration 3d_fullres --fold 0 --epochs 500 --reg_weight 1.0 --reg_loss mse --reg_key bulla_thickness --debug
# 
# 或者使用原始nnUNet命令:
# nnUNetv2_reg_train Dataset102_Reg 3d_fullres 0 -trainer RegnnUNetTrainer_20epochs -reg_weight 1.0 -reg_loss mse -reg_key bulla_thickness 