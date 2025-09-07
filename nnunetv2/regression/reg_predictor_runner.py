import os
import torch
import argparse
import numpy as np
from typing import Union, List, Tuple
from batchgenerators.utilities.file_and_folder_operations import join, isdir, maybe_mkdir_p, subfiles, save_json, isfile, load_json

from nnunetv2.paths import nnUNet_results, nnUNet_raw
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.regression.reg_predictor_fixed import RegnnUNetPredictorFixed


def run_regression_prediction(
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
    
    参数:
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
    
    # 创建预测器 - 使用修复版的RegnnUNetPredictorFixed
    device = torch.device("cuda" if torch.cuda.is_available() and perform_everything_on_gpu else "cpu")
    predictor = RegnnUNetPredictorFixed(
        tile_step_size=tile_step_size,
        use_gaussian=use_gaussian,
        use_mirroring=use_mirroring,
        perform_everything_on_device=perform_everything_on_gpu,
        device=device,
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )
    
    # 确保路径规范化
    model_folder = os.path.abspath(model_folder)
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    
    # 初始化预测器
    try:
        predictor.initialize_from_trained_model_folder(model_folder, use_folds, checkpoint_name)
        if verbose:
            print(f"成功初始化预测器")
    except Exception as e:
        if verbose:
            print(f"初始化失败: {e}")
        raise
    
    # 准备输入文件列表 - 确保文件路径正确
    if isdir(input_folder):
        # 获取文件列表
        file_ending = predictor.dataset_json.get("file_ending", ".nii.gz")
        input_files = subfiles(input_folder, suffix=file_ending, join=False)
        
        # 创建输入文件列表的列表
        list_of_lists = []
        for input_file in input_files:
            file_path = os.path.normpath(join(input_folder, input_file))
            list_of_lists.append([file_path])
    else:
        raise ValueError(f"输入文件夹不存在: {input_folder}")
    
    if verbose:
        print(f"找到 {len(list_of_lists)} 个输入文件")
    
    # 创建输出文件名列表
    output_filenames_truncated = []
    for input_file_list in list_of_lists:
        input_file = input_file_list[0]
        # 获取不包含扩展名的文件名
        filename = os.path.basename(input_file)
        filename_truncated = filename[:-(len(predictor.dataset_json["file_ending"]))]
        # 创建输出文件路径
        output_filename_truncated = os.path.normpath(join(output_folder, filename_truncated))
        output_filenames_truncated.append(output_filename_truncated)
    
    # 创建回归结果文件
    regression_results = {}
    regression_output_file = os.path.normpath(join(output_folder, "regression_results.json"))
    
    # 执行预测 - 使用修复版的 _manage_input_and_output_lists
    list_of_lists, output_filenames_truncated, seg_from_prev_stage_files = \
        predictor._manage_input_and_output_lists(list_of_lists,
                                               output_filenames_truncated,
                                               None, overwrite, 0, 1,
                                               save_probabilities)
    
    # 如果在过滤已存在文件后没有需要处理的文件，直接返回
    if len(list_of_lists) == 0:
        if verbose:
            print("没有需要处理的文件")
        return {}
    
    # 创建数据迭代器
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        list_of_lists,
        seg_from_prev_stage_files,
        output_filenames_truncated,
        num_threads_preprocessing
    )
    
    # 执行预测
    _ = predictor.predict_from_data_iterator(
        data_iterator, 
        save_probabilities, 
        num_threads_nifti_save
    )
    
    # 获取回归结果
    for i, input_file_list in enumerate(list_of_lists):
        try:
            input_file = input_file_list[0]
            # 预处理图像数据，使用专用数据文件名
            preprocessed_data_file = join(output_folder, f'case_{i}_data.npy')
            
            # 检查预处理数据文件是否存在，如果不存在，可能是没有保存或者使用了不同的文件名
            if not isfile(preprocessed_data_file):
                # 尝试使用另一种可能的命名格式
                preprocessed_data_file = join(output_folder, f'data_{i}.npy')
                if not isfile(preprocessed_data_file):
                    # 如果仍找不到文件，跳过这个案例
                    if verbose:
                        print(f"警告: 未找到 {input_file} 的预处理数据文件，跳过回归预测")
                    continue
            
            # 加载预处理数据
            img = torch.from_numpy(np.load(preprocessed_data_file))
            
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
                else:
                    if verbose:
                        print(f"警告: {os.path.basename(input_file)} 没有回归输出")
        except Exception as e:
            if verbose:
                print(f"处理 {input_file_list[0]} 时出错: {e}")
    
    # 保存回归结果
    save_json(regression_results, regression_output_file)
    
    if verbose:
        print(f"回归结果已保存到 {regression_output_file}")
    
    return regression_results


def main():
    parser = argparse.ArgumentParser(description="使用训练好的回归模型进行预测")
    parser.add_argument('-i', type=str, required=True,
                        help='输入文件夹，包含要预测的图像')
    parser.add_argument('-o', type=str, required=True,
                        help='输出文件夹，保存预测结果')
    parser.add_argument('-m', type=str, required=False, default=None,
                        help='模型文件夹，包含训练好的模型。如果不提供，将使用-d,-tr,-p,-c参数')
    parser.add_argument('-d', type=str, required=False, default=None,
                        help='数据集名称或ID，与-m互斥')
    parser.add_argument('-tr', type=str, required=False, default="nnUNetTrainer",
                        help='训练器名称，例如RegnnUNetTrainer，与-m互斥')
    parser.add_argument('-p', type=str, required=False, default="nnUNetPlans",
                        help='计划标识符，与-m互斥')
    parser.add_argument('-c', type=str, required=False, default="3d_fullres",
                        help='配置名称，与-m互斥')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=None,
                        help='要使用的折叠，例如0 1 2 3 4')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='滑动窗口步长，默认0.5')
    parser.add_argument('--disable_tta', action='store_true', required=False,
                        help='禁用测试时数据增强（镜像）')
    parser.add_argument('--verbose', action='store_true', required=False, default=True,
                        help='详细输出')
    parser.add_argument('--save_probabilities', action='store_true', required=False,
                        help='保存预测的概率图')
    parser.add_argument('--continue_prediction', action='store_true', required=False,
                        help='继续中断的预测（不覆盖现有文件）')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='要使用的检查点名称，默认checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=4,
                        help='预处理使用的进程数，默认4')
    parser.add_argument('-nps', type=int, required=False, default=2,
                        help='分割导出使用的进程数，默认2')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help='使用的设备，可选cuda（GPU）或cpu')
    
    args = parser.parse_args()
    
    # 处理互斥参数
    if args.m is None and args.d is None:
        parser.error("必须提供-m或-d参数")
    
    # 获取模型文件夹
    if args.m is not None:
        model_folder = args.m
    else:
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        model_folder = get_output_folder(args.d, args.tr, args.p, args.c)
    
    # 处理折叠参数
    if args.f is not None:
        use_folds = [int(i) if i != 'all' else i for i in args.f]
    else:
        use_folds = None
    
    # 根据设备设置GPU使用
    perform_everything_on_gpu = (args.device.lower() == 'cuda')
    
    print(f"使用模型: {model_folder}")
    print(f"输入文件夹: {args.i}")
    print(f"输出文件夹: {args.o}")
    
    # 执行预测
    run_regression_prediction(
        args.i,
        args.o,
        model_folder,
        use_folds,
        args.step_size,
        not args.disable_tta,  # use_mirroring
        True,  # use_gaussian
        perform_everything_on_gpu,
        args.verbose,
        args.save_probabilities,
        not args.continue_prediction,  # overwrite
        args.chk,
        args.npp,
        args.nps
    )


if __name__ == "__main__":
    main() 