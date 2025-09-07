import torch
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerDualDecoder(nnUNetTrainer):
    """
    专门用于DualDecoderUNet的训练器
    支持双解码器输出：分割和回归
    """
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> 'DualDecoderUNet':
        """
        构建DualDecoderUNet网络架构
        """
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )
    
    def train_step(self, batch: dict) -> dict:
        """
        重写训练步骤以处理双解码器输出
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            
            # 处理双解码器输出
            if isinstance(output, (tuple, list)) and len(output) == 2:
                seg_output, reg_output = output
            elif isinstance(output, dict):
                seg_output = output['segmentation']
                reg_output = output.get('regression', None)
            else:
                seg_output = output
                reg_output = None
            
            # 计算分割损失
            seg_loss = self.loss(seg_output, target)
            total_loss = seg_loss
            
            # 如果有回归输出和回归目标，计算回归损失
            if reg_output is not None and 'regression_target' in batch:
                reg_target = batch['regression_target'].to(self.device, non_blocking=True)
                reg_loss = torch.nn.functional.mse_loss(reg_output, reg_target)
                
                # 获取回归损失权重
                reg_weight = getattr(self.network, 'regression_loss_weight', 0.1)
                total_loss = seg_loss + reg_weight * reg_loss
                
                return {
                    'loss': total_loss,
                    'seg_loss': seg_loss,
                    'reg_loss': reg_loss,
                    'reg_weight': reg_weight
                }
        
        return {'loss': total_loss, 'seg_loss': seg_loss}
    
    def validation_step(self, batch: dict) -> dict:
        """
        重写验证步骤以处理双解码器输出
        """
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                output = self.network(data)
                
                # 处理双解码器输出
                if isinstance(output, (tuple, list)) and len(output) == 2:
                    seg_output, reg_output = output
                elif isinstance(output, dict):
                    seg_output = output['segmentation']
                    reg_output = output.get('regression', None)
                else:
                    seg_output = output
                    reg_output = None
                
                # 计算分割损失
                seg_loss = self.loss(seg_output, target)
                total_loss = seg_loss
                
                # 如果有回归输出和回归目标，计算回归损失
                if reg_output is not None and 'regression_target' in batch:
                    reg_target = batch['regression_target'].to(self.device, non_blocking=True)
                    reg_loss = torch.nn.functional.mse_loss(reg_output, reg_target)
                    
                    # 获取回归损失权重
                    reg_weight = getattr(self.network, 'regression_loss_weight', 0.1)
                    total_loss = seg_loss + reg_weight * reg_loss
                    
                    return {
                        'loss': total_loss,
                        'seg_loss': seg_loss,
                        'reg_loss': reg_loss,
                        'reg_weight': reg_weight
                    }
        
        return {'loss': total_loss, 'seg_loss': seg_loss}


class nnUNetTrainerDualDecoder_5epochs(nnUNetTrainerDualDecoder):
    """
    5个epoch的DualDecoder训练器，用于快速测试
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5


class nnUNetTrainerDualDecoder_10epochs(nnUNetTrainerDualDecoder):
    """
    10个epoch的DualDecoder训练器，用于初步训练
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10


class nnUNetTrainerDualDecoder_50epochs(nnUNetTrainerDualDecoder):
    """
    50个epoch的DualDecoder训练器，用于较短的训练
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 50