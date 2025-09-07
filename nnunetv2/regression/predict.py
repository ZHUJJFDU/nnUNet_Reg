#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
import numpy as np
from typing import Union, List, Tuple, Dict
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, save_json, isfile

from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.regression.reg_predictor import RegnnUNetPredictor


def predict_regression_from_raw_data(
    input_folder: str,
    output_folder: str,
    model_folder: str,
    use_folds: Union[List[int], Tuple[int, ...]] = None,
    tile_step_size: float = 0.5,
    use_gaussian: bool = True,
    use_mirroring: bool = True,
    perform_everything_on_gpu: bool = True,
    verbose: bool = True,
    save_probabilities: bool = False,
    overwrite: bool = True,
    checkpoint_name: str = "checkpoint_final.pth",
    num_threads_preprocessing: int = 8,
    num_threads_nifti_save: int = 2,
):
    """
    使用训练好的回归模型进行预测

    Args:
        input_folder: 包含要预测的图像的文件夹
        output_folder: 保存预测结果的文件夹
        model_folder: 包含模型的文件夹
        use_folds: 要使用的折叠，如果为None，则使用所有可用的折叠
        tile_step_size: 滑动窗口步长
        use_gaussian: 是否使用高斯加权
        use_mirroring: 是否使用镜像增强
        perform_everything_on_gpu: 是否在GPU上执行所有操作
        verbose: 是否打印详细信息
        save_probabilities: 是否保存概率图
        overwrite: 是否覆盖现有的预测结果
        checkpoint_name: 要使用的检查点名称
        num_threads_preprocessing: 预处理线程数
        num_threads_nifti_save: 保存nifti文件的线程数
    """
    # 创建输出文件夹
    maybe_mkdir_p(output_folder)
    
    if verbose:
        print(f"模型文件夹: {model_folder}")
        if isfile(join(model_folder, 'dataset.json')):
            print(f"找到dataset.json: {join(model_folder, 'dataset.json')}")
        else:
            print(f"未找到dataset.json: {join(model_folder, 'dataset.json')}")
    
    # 创建预测器
    predictor = RegnnUNetPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_gpu,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )
    
    # 使用标准的初始化方法
    try:
        # 尝试从模型文件夹初始化
        predictor.initialize_from_trained_model_folder(model_folder, use_folds, checkpoint_name)
        if verbose:
            print(f"成功使用标准初始化方法")
    except FileNotFoundError as e:
        # 如果dataset.json不在fold_0文件夹中，尝试在主文件夹中查找
        if verbose:
            print(f"标准初始化失败: {e}")
            print("尝试在主文件夹中查找配置文件...")
        
        # 检查主文件夹中是否有dataset.json和plans.json
        dataset_json_path = join(model_folder, 'dataset.json')
        plans_file_path = join(model_folder, 'plans.json')
        
        if not isfile(dataset_json_path) or not isfile(plans_file_path):
            raise FileNotFoundError(f"未找到必要的配置文件。请确保{dataset_json_path}和{plans_file_path}存在。")
        
        if verbose:
            print(f"找到配置文件，手动初始化预测器...")
        
        # 手动初始化
        from batchgenerators.utilities.file_and_folder_operations import load_json
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.inference.data_iterators import PreprocessAdapter
        
        # 加载配置文件
        dataset_json = load_json(dataset_json_path)
        plans = load_json(plans_file_path)
        plans_manager = PlansManager(plans)
        
        # 获取配置
        if 'nnUNetPlans' in model_folder:
            configuration_name = model_folder.split('__')[-1]
            configuration_manager = plans_manager.get_configuration(configuration_name)
        else:
            configuration_manager = plans_manager.get_configuration('3d_fullres')
        
        # 获取标签管理器
        label_manager = plans_manager.get_label_manager(dataset_json)
        
        # 获取网络架构类名和参数
        configuration_name = '3d_fullres'
        if 'nnUNetPlans' in model_folder:
            configuration_name = model_folder.split('__')[-1]
        
        # 确定输入通道数
        from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        
        # 导入nnUNetTrainer类
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        
        # 构建网络
        network = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False
        )
        
        # 设置预测器属性
        predictor.plans_manager = plans_manager
        predictor.configuration_manager = configuration_manager
        predictor.network = network
        predictor.dataset_json = dataset_json
        predictor.label_manager = label_manager
        predictor.network_config = None  # 添加network_config属性
        
        # 加载权重
        if use_folds is None:
            # 查找所有可用的折叠
            fold_folders = [i for i in os.listdir(model_folder) if i.startswith('fold_') and
                           isdir(join(model_folder, i))]
            use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        
        if verbose:
            print(f"使用折叠: {use_folds}")
        
        # 加载权重
        predictor.list_of_parameters = []
        for fold in use_folds:
            fold_dir = join(model_folder, f'fold_{fold}')
            if not isdir(fold_dir):
                raise FileNotFoundError(f"未找到折叠目录: {fold_dir}")
                
            checkpoint_file = join(fold_dir, checkpoint_name)
            
            if not isfile(checkpoint_file):
                raise FileNotFoundError(f"未找到检查点文件: {checkpoint_file}")
            
            if verbose:
                print(f"加载检查点: {checkpoint_file}")
            
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'network_weights' in checkpoint:
                predictor.list_of_parameters.append(checkpoint['network_weights'])
            else:
                predictor.list_of_parameters.append(checkpoint)
        
        # 创建预处理器
        predictor.preprocessor = PreprocessAdapter(configuration_manager, plans_manager,
                                             dataset_json, predictor.device)
    
    # 获取输入文件列表
    if isdir(input_folder):
        # 获取完整路径
        input_files = subfiles(input_folder, suffix=predictor.dataset_json["file_ending"], join=True)
        # 确保路径格式正确（特别是在Windows系统上）
        input_files = [os.path.normpath(i) for i in input_files]
        # 获取文件名（不包含路径）
        input_file_names = [os.path.basename(i) for i in input_files]
    else:
        # 单个文件
        input_files = [os.path.normpath(input_folder)]
        input_file_names = [os.path.basename(input_folder)]
    
    if verbose:
        print(f"找到 {len(input_files)} 个输入文件")
    
    # 创建输出文件
    output_files = []
    for i in input_file_names:
        output_file = os.path.normpath(join(output_folder, i))
        if output_file.endswith(predictor.dataset_json["file_ending"]):
            output_file = output_file[:-len(predictor.dataset_json["file_ending"])]
        output_files.append(output_file)
    
    # 创建回归结果文件
    regression_results = {}
    regression_output_file = os.path.normpath(join(output_folder, "regression_results.json"))
    
    # 执行预测
    predictor.predict_from_files(
        input_files,
        output_files,
        save_probabilities,
        overwrite,
        num_threads_preprocessing,
        num_threads_nifti_save,
        None,  # folder_with_segs_from_prev_stage
        1,     # num_parts
        0      # part_id
    )
    
    # 获取回归结果
    for i, input_file in enumerate(input_files):
        try:
            # 读取图像数据
            img = predictor.preprocess_input_file(input_file)
            
            # 进行预测
            with torch.no_grad():
                result = predictor.predict_sliding_window_return_logits(img)
                
                # 检查是否有回归输出
                if isinstance(result, tuple) and len(result) == 2:
                    _, regression = result
                    # 获取回归值
                    regression_value = float(regression.cpu().numpy().mean())
                    regression_results[os.path.basename(input_file)] = regression_value
                    
                    if verbose:
                        print(f"回归预测 {os.path.basename(input_file)}: {regression_value:.4f}")
        except Exception as e:
            if verbose:
                print(f"处理 {input_file} 时出错: {e}")
    
    # 保存回归结果
    save_json(regression_results, regression_output_file)
    
    if verbose:
        print(f"回归结果已保存到 {regression_output_file}")
    
    return regression_results


def main():
    parser = argparse.ArgumentParser(description="使用训练好的回归模型进行预测")
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="输入文件夹")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="输出文件夹")
    parser.add_argument("-m", "--model_folder", type=str, required=True,
                        help="模型文件夹（必须使用RegnnUNetTrainer训练）")
    parser.add_argument("-f", "--folds", nargs="+", type=int, default=None,
                        help="要使用的折叠（默认：全部）")
    parser.add_argument("-s", "--step_size", type=float, default=0.5,
                        help="滑动窗口步长（默认：0.5）")
    parser.add_argument("-g", "--use_gaussian", action="store_true", default=True,
                        help="使用高斯加权（默认：是）")
    parser.add_argument("--no_gaussian", action="store_false", dest="use_gaussian",
                        help="不使用高斯加权")
    parser.add_argument("-r", "--use_mirroring", action="store_true", default=True,
                        help="使用镜像增强（默认：是）")
    parser.add_argument("--no_mirroring", action="store_false", dest="use_mirroring",
                        help="不使用镜像增强")
    parser.add_argument("--gpu", action="store_true", default=True,
                        help="在GPU上执行所有操作（默认：是）")
    parser.add_argument("--cpu", action="store_false", dest="gpu",
                        help="不在GPU上执行所有操作")
    parser.add_argument("-v", "--verbose", action="store_true", default=True,
                        help="详细输出（默认：是）")
    parser.add_argument("--quiet", action="store_false", dest="verbose",
                        help="简洁输出")
    parser.add_argument("-p", "--save_probabilities", action="store_true", default=False,
                        help="保存概率图（默认：否）")
    parser.add_argument("--overwrite", action="store_true", default=True,
                        help="覆盖现有预测结果（默认：是）")
    parser.add_argument("--no_overwrite", action="store_false", dest="overwrite",
                        help="不覆盖现有预测结果")
    parser.add_argument("-c", "--checkpoint", type=str, default="checkpoint_final.pth",
                        help="检查点名称（默认：checkpoint_final.pth）")
    parser.add_argument("--threads_preprocessing", type=int, default=8,
                        help="预处理线程数（默认：8）")
    parser.add_argument("--threads_save", type=int, default=2,
                        help="保存线程数（默认：2）")
    
    args = parser.parse_args()
    
    # 执行预测
    predict_regression_from_raw_data(
        args.input_folder,
        args.output_folder,
        args.model_folder,
        args.folds,
        args.step_size,
        args.use_gaussian,
        args.use_mirroring,
        args.gpu,
        args.verbose,
        args.save_probabilities,
        args.overwrite,
        args.checkpoint,
        args.threads_preprocessing,
        args.threads_save
    )


if __name__ == "__main__":
    main() 