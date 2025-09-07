import torch

from nnunetv2.training.nnUNetTrainer.RegnnUNetTrainer import DualRegnnUNetTrainer


class DualRegnnUNetTrainer_5epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """用于快速测试的5轮训练版本"""
        num_epochs = 5  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_10epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """用于快速测试的10轮训练版本"""
        num_epochs = 10  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_20epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """用于快速测试的20轮训练版本"""
        num_epochs = 20  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_50epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """用于快速收敛的50轮训练版本"""
        num_epochs = 50  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_100epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """标准训练轮数"""
        num_epochs = 100  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_250epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """延长训练轮数"""
        num_epochs = 250  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_500epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """长训练轮数"""
        num_epochs = 500  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_1000epochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """非常长的训练轮数"""
        num_epochs = 1000  # 在调用super().__init__()之前设置
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler()


class DualRegnnUNetTrainer_CustomEpochs(DualRegnnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'), num_epochs: int = 1000):
        """可自定义训练轮数的版本"""
        # num_epochs already available as parameter
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = num_epochs
        # 确保amp属性存在
        if not hasattr(self, 'amp'):
            self.amp = True
        # 确保amp_grad_scaler属性存在
        if self.amp and not hasattr(self, 'amp_grad_scaler'):
            from torch.cuda.amp import GradScaler
            self.amp_grad_scaler = GradScaler() 