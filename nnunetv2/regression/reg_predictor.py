import torch
import numpy as np
from typing import Union, Tuple, List
from copy import deepcopy
import itertools
from contextlib import contextmanager
import os

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from tqdm import tqdm


@contextmanager
def dummy_context():
    yield None


class RegnnUNetPredictor(nnUNetPredictor):
    """
    扩展nnUNetPredictor，支持回归输出
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 回归输出标志
        self.return_regression = kwargs.pop('return_regression', True)
        
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        重写镜像预测方法，支持回归输出
        """
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)
        
        # 检查网络输出是否包含回归值
        if isinstance(prediction, tuple) and len(prediction) == 2:
            seg_output, reg_output = prediction
            has_regression = True
        else:
            seg_output = prediction
            reg_output = None
            has_regression = False
        
        if mirror_axes is not None:
            # 检查mirror_axes中的无效数字
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            
            for axes in axes_combinations:
                # 对输入进行镜像，获取预测结果
                flipped_x = torch.flip(x, axes)
                mirrored_prediction = self.network(flipped_x)
                
                # 处理回归输出
                if isinstance(mirrored_prediction, tuple) and len(mirrored_prediction) == 2:
                    mirrored_seg, mirrored_reg = mirrored_prediction
                    # 对分割结果进行镜像还原
                    mirrored_seg = torch.flip(mirrored_seg, axes)
                    # 回归值不需要镜像
                    if has_regression:
                        seg_output += mirrored_seg
                        reg_output += mirrored_reg
                    else:
                        seg_output += mirrored_seg
                        has_regression = True
                        reg_output = mirrored_reg
                else:
                    # 只有分割结果
                    mirrored_seg = torch.flip(mirrored_prediction, axes)
                    seg_output += mirrored_seg
            
            # 计算平均值
            seg_output /= (len(axes_combinations) + 1)
            if has_regression and reg_output is not None:
                reg_output /= (len(axes_combinations) + 1)
        
        # 根据是否有回归输出返回不同的结果
        if has_regression and reg_output is not None:
            return seg_output, reg_output
        else:
            return seg_output
    
    def _internal_predict_sliding_window_return_logits(self,
                                                      data: torch.Tensor,
                                                      slicers,
                                                      do_on_device: bool = True,
                                                      ):
        """
        重写滑动窗口预测方法，支持回归输出
        """
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        predicted_regression = None
        results_device = self.device if do_on_device else torch.device('cpu')
        has_regression_output = False

        try:
            empty_cache(self.device)

            # 将数据移动到设备
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # 预分配数组
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                         dtype=torch.half,
                                         device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                          value_scaling_factor=10,
                                          device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
                
            # 用于累积回归预测的列表
            regression_predictions = []
            
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                # 获取预测结果
                prediction_result = self._internal_maybe_mirror_and_predict(workon)
                
                # 检查是否有回归输出
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    prediction, regression = prediction_result
                    regression = regression.to('cpu')  # 移到CPU以避免GPU内存问题
                    
                    # 收集回归预测
                    regression_predictions.append(regression.detach())
                    has_regression_output = True
                else:
                    prediction = prediction_result
                
                prediction = prediction.to(results_device)
                
                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            
            # 如果有回归输出，计算平均值
            if has_regression_output and len(regression_predictions) > 0:
                # 将所有回归预测堆叠并计算平均值
                predicted_regression = torch.mean(torch.cat(regression_predictions, dim=0), dim=0, keepdim=True)
            
            # 检查无穷大值
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                 'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                 'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            if has_regression_output and len(regression_predictions) > 0:
                del regression_predictions
                if predicted_regression is not None:
                    del predicted_regression
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        
        # 根据是否有回归输出返回不同的结果
        if has_regression_output and predicted_regression is not None:
            return predicted_logits, predicted_regression
        else:
            return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        使用滑动窗口进行预测，返回分割logits和回归值（如果有）
        
        Args:
            input_image: 输入图像张量，形状为(c, x, y, z)
            
        Returns:
            如果有回归输出，返回(predicted_logits, predicted_regression)元组
            否则只返回predicted_logits
        """
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # 自动混合精度设置
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # 如果输入图像小于tile_size，需要填充
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                         'constant', {'value': 0}, True,
                                                         None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # 尝试在设备上进行预测，如果内存不足则回退到CPU
                    try:
                        prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                         self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    prediction_result = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                     self.perform_everything_on_device)

                empty_cache(self.device)
                
                # 检查结果类型并相应地恢复填充
                if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                    predicted_logits, predicted_regression = prediction_result
                    # 恢复填充（只对分割结果）
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                    return predicted_logits, predicted_regression
                else:
                    # 只有分割结果
                    predicted_logits = prediction_result
                    # 恢复填充
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                    return predicted_logits 

    def preprocess_input_file(self, input_file: str) -> torch.Tensor:
        """
        预处理输入文件，返回可用于预测的张量
        
        Args:
            input_file: 输入文件路径
            
        Returns:
            预处理后的输入张量
        """
        # 读取图像数据
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        
        # 使用配置中的图像读取器
        reader = self.plans_manager.image_reader_writer_class()
        
        # 读取图像
        image, properties = reader.read_images([input_file])
        
        # 预处理图像
        data, properties = self.preprocessor.run_case(image, properties, self.plans_manager, self.configuration_manager,
                                                    self.dataset_json)
        
        # 转换为PyTorch张量
        data = torch.from_numpy(data)
        
        return data
        
    def initialize_from_trained_model_folder_custom(self, model_training_output_dir: str,
                                             use_folds: Union[List[int], Tuple[int, ...]] = None,
                                             checkpoint_name: str = "checkpoint_final.pth"):
        """
        自定义初始化方法，用于处理正确的模型文件夹结构
        
        Args:
            model_training_output_dir: 模型训练输出目录
            use_folds: 要使用的折叠，如果为None，则使用所有可用的折叠
            checkpoint_name: 要使用的检查点名称
        """
        from batchgenerators.utilities.file_and_folder_operations import join, isdir, isfile, load_json
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
        
        # 检查dataset.json是否在主文件夹中而不是fold_0中
        if isfile(join(model_training_output_dir, 'dataset.json')):
            dataset_json_path = join(model_training_output_dir, 'dataset.json')
            plans_file_path = join(model_training_output_dir, 'plans.json')
        else:
            # 尝试在fold_0文件夹中查找
            fold_dir = join(model_training_output_dir, 'fold_0')
            if isfile(join(fold_dir, 'dataset.json')):
                dataset_json_path = join(fold_dir, 'dataset.json')
                plans_file_path = join(fold_dir, 'plans.json')
            else:
                raise FileNotFoundError(f"Could not find dataset.json in {model_training_output_dir} or {fold_dir}")
        
        # 加载dataset.json
        self.dataset_json = load_json(dataset_json_path)
        
        # 加载plans.json
        plans = load_json(plans_file_path)
        self.plans_manager = PlansManager(plans)
        
        # 获取配置
        if 'nnUNetPlans' in model_training_output_dir:
            configuration_name = model_training_output_dir.split('__')[-1]
            self.configuration_manager = self.plans_manager.get_configuration(configuration_name)
        else:
            self.configuration_manager = self.plans_manager.get_configuration('3d_fullres')
        
        # 获取标签管理器 - 使用plans_manager的方法
        self.label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        
        # 创建网络
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        # 从配置中获取网络类名
        configuration_name = '3d_fullres'
        if 'nnUNetPlans' in model_training_output_dir:
            configuration_name = model_training_output_dir.split('__')[-1]
        network_class_name = self.plans_manager.plans['configurations'][configuration_name]['architecture']['network_class_name']
        
        self.network = nnUNetTrainer.build_network_architecture(
            network_class_name,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            self.label_manager,
            self.network_config
        )
        
        # 加载权重
        if use_folds is None:
            # 查找所有可用的折叠
            fold_folders = [i for i in os.listdir(model_training_output_dir) if i.startswith('fold_') and
                           isdir(join(model_training_output_dir, i))]
            use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        
        # 加载权重
        self.list_of_parameters = []
        for fold in use_folds:
            checkpoint_file = join(model_training_output_dir, f'fold_{fold}', checkpoint_name)
            if not isfile(checkpoint_file):
                # 尝试在主文件夹中查找
                checkpoint_file = join(model_training_output_dir, checkpoint_name)
                if not isfile(checkpoint_file):
                    raise FileNotFoundError(f"Could not find checkpoint {checkpoint_name} for fold {fold}")
            
            self.print_to_log_file(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            if 'network_weights' in checkpoint:
                self.list_of_parameters.append(checkpoint['network_weights'])
            else:
                self.list_of_parameters.append(checkpoint)
        
        # 创建预处理器
        from nnunetv2.inference.data_iterators import PreprocessAdapter
        self.preprocessor = PreprocessAdapter(self.configuration_manager, self.plans_manager,
                                             self.dataset_json, self.device) 