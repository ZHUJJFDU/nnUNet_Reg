import torch
from nnunetv2.training.nnUNetTrainer.RegnnUNetTrainer import RegnnUNetTrainer


class RegnnUNetTrainer_CPU(RegnnUNetTrainer):
    """
    CPU-only version of RegnnUNetTrainer for debugging purposes.
    This trainer forces the use of CPU even if CUDA is available.
    """
    
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True, device=None):
        """
        Initialize with CPU device regardless of what is passed
        """
        # Force CPU device
        device = torch.device('cpu')
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file("Using CPU-only trainer for debugging")
    
    def _build_loss_function(self):
        """
        Build the loss function with simpler settings for CPU
        """
        super()._build_loss_function()
        self.print_to_log_file("Using simplified loss function for CPU")
    
    def _build_network_architecture(self):
        """
        Build a smaller network for CPU training
        """
        # Call the parent method to build the network
        network = super()._build_network_architecture()
        
        # Print a warning about slow CPU training
        self.print_to_log_file("WARNING: Training on CPU will be significantly slower than on GPU")
        self.print_to_log_file("This trainer is intended for debugging purposes only")
        
        return network
    
    def train_step(self, batch):
        """
        Simplified training step for CPU
        """
        # Disable timing for CPU version to avoid overhead
        old_enable_timing = self.enable_timing
        self.enable_timing = False
        
        # Call parent train_step
        result = super().train_step(batch)
        
        # Restore timing setting
        self.enable_timing = old_enable_timing
        
        return result
    
    def validation_step(self, batch):
        """
        Simplified validation step for CPU
        """
        # Disable timing for CPU version to avoid overhead
        old_enable_timing = self.enable_timing
        self.enable_timing = False
        
        # Call parent validation_step
        result = super().validation_step(batch)
        
        # Restore timing setting
        self.enable_timing = old_enable_timing
        
        return result
    
    def run_training(self):
        """
        Run training with reduced batch size and iterations for CPU
        """
        # Reduce batch size for CPU training
        original_batch_size = self.batch_size
        self.batch_size = max(1, self.batch_size // 4)
        self.print_to_log_file(f"Reduced batch size from {original_batch_size} to {self.batch_size} for CPU training")
        
        # Reduce number of iterations for faster debugging
        original_iterations = self.num_iterations_per_epoch
        self.num_iterations_per_epoch = min(10, self.num_iterations_per_epoch)
        self.print_to_log_file(f"Reduced iterations from {original_iterations} to {self.num_iterations_per_epoch} for debugging")
        
        # Reduce validation iterations as well
        original_val_iterations = self.num_val_iterations_per_epoch
        self.num_val_iterations_per_epoch = min(5, self.num_val_iterations_per_epoch)
        self.print_to_log_file(f"Reduced validation iterations from {original_val_iterations} to {self.num_val_iterations_per_epoch} for debugging")
        
        # Call parent run_training
        super().run_training() 