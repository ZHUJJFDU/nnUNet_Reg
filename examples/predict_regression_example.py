#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nnUNet回归模型预测示例脚本
"""

import os
import argparse
import sys
from nnunetv2.regression.reg_predictor_runner import run_regression_prediction
from nnunetv2.utilities.file_path_utilities import get_output_folder
from batchgenerators.utilities.file_and_folder_operations import isdir, maybe_mkdir_p


def main():
    parser = argparse.ArgumentParser(description="nnUNet回归模型预测示例")
    parser.add_argument('-i', type=str, required=True,
                      help='输入文件夹，包含待预测的图像')
    parser.add_argument('-o', type=str, required=True,
                      help='输出文件夹，用于保存预测结果')
    parser.add_argument('-d', type=str, required=True,
                      help='数据集ID或名称，例如"102"')
    parser.add_argument('-t', type=str, required=False, default="nnUNetTrainer",
                      help='训练器名称，例如"RegnnUNetTrainer_5epochs"')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=['0'],
                      help='要使用的折叠，例如0 1 2 3 4')
    parser.add_argument('-npp', type=int, required=False, default=2,
                      help='预处理线程数，默认为2')
    parser.add_argument('-nps', type=int, required=False, default=2,
                      help='分割导出线程数，默认为2')
    
    args = parser.parse_args()
    
    # 确保路径是绝对路径并且规范化
    input_folder = os.path.abspath(os.path.normpath(args.i))
    output_folder = os.path.abspath(os.path.normpath(args.o))
    
    # 检查输入文件夹是否存在
    if not isdir(input_folder):
        print(f"错误: 输入文件夹不存在: {input_folder}")
        sys.exit(1)
    
    # 创建输出文件夹
    maybe_mkdir_p(output_folder)
    
    try:
        # 获取模型文件夹
        try:
            model_folder = get_output_folder(args.d, args.t, "nnUNetPlans", "3d_fullres")
            if not isdir(model_folder):
                raise FileNotFoundError(f"模型文件夹不存在: {model_folder}")
        except Exception as e:
            # 尝试使用自定义路径
            dataset_name = f"Dataset{args.d}_Reg"
            model_folder = os.path.normpath(f"DATASET/nnUNet_trained_models/{dataset_name}/{args.t}__nnUNetPlans__3d_fullres")
            if not isdir(model_folder):
                print(f"错误: 无法找到模型文件夹: {e}")
                print(f"尝试使用: {model_folder}")
                print("请确保模型文件夹路径正确")
                sys.exit(1)
        
        print(f"使用模型: {model_folder}")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        
        # 执行预测
        run_regression_prediction(
            input_folder=input_folder,
            output_folder=output_folder,
            model_folder=model_folder,
            use_folds=[int(f) for f in args.f],
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            verbose=True,
            save_probabilities=False,
            overwrite=True,
            checkpoint_name="checkpoint_final.pth",
            num_threads_preprocessing=args.npp,
            num_threads_nifti_save=args.nps
        )
        
        print("预测完成！")
        
    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 