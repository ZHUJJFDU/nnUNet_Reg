from typing import Union, List, Tuple

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.regression.reg_dataset import RegnnUNetDataset


class RegnnUNetDataLoader(nnUNetDataLoaderBase):
    """
    Data loader for regression with nnUNet
    Extends the base nnUNet data loader to include regression values
    """
    def __init__(self, data: RegnnUNetDataset, batch_size: int, patch_size: Union[List[int], Tuple[int, ...]], 
                 final_patch_size: Union[List[int], Tuple[int, ...]], label_manager, oversample_foreground_percent: float = 0.0,
                 memmap_mode: str = "r", num_threads_in_multithreaded: int = 1, shuffle: bool = True, 
                 verbose: bool = False):
        """
        Initialize RegnnUNetDataLoader

        Args:
            data: RegnnUNetDataset instance
            batch_size: Batch size for training
            patch_size: Size of the patches to extract
            final_patch_size: Final size of the patches (after resampling)
            label_manager: Label manager for handling segmentation labels
            oversample_foreground_percent: Percentage of batch to be foreground patches
            memmap_mode: Mode for numpy memory mapping
            num_threads_in_multithreaded: Number of threads for multithreaded loading
            shuffle: Whether to shuffle the data
            verbose: Whether to print debug information
        """
        # Store dataset for later reference (needed for overriding methods)
        self._orig_data = data
        
        # Create label transform to handle -1 values
        self.label_transform = RemoveLabelTansform(-1, 0)
        
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                        oversample_foreground_percent, memmap_mode, num_threads_in_multithreaded,
                        shuffle, verbose)
    
    def determine_shapes(self):
        """
        Override determine_shapes to handle the 4-value return from RegnnUNetDataset.load_case
        
        Returns:
            Tuple of (data_shape, seg_shape) for batch construction
        """
        # Load one case to determine shapes, but handle the 4-value return
        # RegnnUNetDataset.load_case returns (data, seg, properties, regression_value)
        data, seg, _, _ = self._orig_data.load_case(self.indices[0])
        num_color_channels = data.shape[0]

        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, seg.shape[0], *self.patch_size)
        return data_shape, seg_shape
    
    def generate_train_batch(self):
        """
        Generate a training batch with regression values
        
        Returns:
            dict: Training batch with data, target, seg, regression_value, and other properties
        """
        # Reimplementing the _get_random_samples functionality directly
        selected_keys = []
        # preallocate memory for data and seg
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.int16)
        
        # Create array for regression values with batch size
        regression_values = np.zeros((self.batch_size, 1), dtype=np.float32)
        
        # Track properties for random crop if needed
        properties_for_random_crop = []
        
        # Process each case
        for i in range(self.batch_size):
            # decide if we need to use the oversample method or not
            force_fg = self.get_do_oversample(i)
            
            # randomly select a key
            if force_fg and not self.has_ignore:
                # filter out cases that don't have foreground
                eligible_keys_for_fg = [i for i in self.indices if
                                        self.annotated_classes_key in self._orig_data[i]['properties']['class_locations'].keys()]
                
                # if no foreground cases are available, fall back to random
                if len(eligible_keys_for_fg) == 0:
                    key = self.indices[np.random.choice(len(self.indices))]
                else:
                    key = eligible_keys_for_fg[np.random.choice(len(eligible_keys_for_fg))]
            else:
                key = self.indices[np.random.choice(len(self.indices))]
            
            # Load case with regression value - RegnnUNetDataset.load_case returns 4 values
            case_data, case_seg, case_props, regression_value = self._orig_data.load_case(key)
            
            # Apply RemoveLabelTansform to convert -1 to 0 in segmentation
            case_seg = self._apply_label_transform(case_seg)
            
            # extract random data and seg patch
            if not force_fg and not self.has_ignore:
                # get a random patch
                bbox_lbs, bbox_ubs = self.get_bbox(case_data.shape[1:], force_fg=False, 
                                                  class_locations=None)
            else:
                # filter out classes that don't exist
                class_locations = {c: l for c, l in case_props['class_locations'].items() if len(l) > 0}
                
                # get a random patch with foreground 
                bbox_lbs, bbox_ubs = self.get_bbox(case_data.shape[1:], force_fg=True,
                                                 class_locations=class_locations)
            
            # extract the patch
            # Extract the bounding box with padding/cropping to ensure it matches the expected patch size
            data_shape_each = case_data.shape[1:]
            
            # Make sure we don't go out of bounds
            bbox_lbs = [max(0, bbox_lb) for bbox_lb in bbox_lbs]
            bbox_ubs = [min(shape_dim, bbox_ub) for shape_dim, bbox_ub in zip(data_shape_each, bbox_ubs)]
            
            # Get the actual data and seg patches
            data_patch = case_data[:, bbox_lbs[0]:bbox_ubs[0], bbox_lbs[1]:bbox_ubs[1], bbox_lbs[2]:bbox_ubs[2]]
            seg_patch = case_seg[:, bbox_lbs[0]:bbox_ubs[0], bbox_lbs[1]:bbox_ubs[1], bbox_lbs[2]:bbox_ubs[2]]
            
            # Get the actual shape of the patches
            data_patch_shape = data_patch.shape[1:]
            
            # Calculate padding needed on each side
            pad_before = [0, 0, 0]  # No padding needed for channel dimension
            pad_after = [0, 0, 0]   # No padding needed for channel dimension
            
            for dim in range(3):  # For 3D data
                # If patch is smaller than needed, pad it
                if data_patch_shape[dim] < self.patch_size[dim]:
                    # Calculate padding for each side
                    total_pad = self.patch_size[dim] - data_patch_shape[dim]
                    pad_before[dim] = total_pad // 2
                    pad_after[dim] = total_pad - pad_before[dim]
                # If patch is larger than needed, crop it
                elif data_patch_shape[dim] > self.patch_size[dim]:
                    # Calculate cropping for each side
                    total_crop = data_patch_shape[dim] - self.patch_size[dim]
                    pad_before[dim] = -total_crop // 2
                    pad_after[dim] = -total_crop + pad_before[dim]
            
            # Create the final patches with correct shape
            final_data_patch = np.zeros((data_patch.shape[0], *self.patch_size), dtype=np.float32)
            final_seg_patch = np.zeros((seg_patch.shape[0], *self.patch_size), dtype=np.int16)
            
            # Define the slices for the source and destination
            src_slices = []
            dst_slices = []
            
            for dim in range(3):
                if pad_before[dim] < 0:  # Need to crop
                    src_slices.append(slice(-pad_before[dim], data_patch_shape[dim] + pad_after[dim]))
                    dst_slices.append(slice(0, self.patch_size[dim]))
                else:  # Need to pad
                    src_slices.append(slice(0, data_patch_shape[dim]))
                    dst_slices.append(slice(pad_before[dim], pad_before[dim] + data_patch_shape[dim]))
            
            # Copy the data
            final_data_patch[:, dst_slices[0], dst_slices[1], dst_slices[2]] = data_patch[:, src_slices[0], src_slices[1], src_slices[2]]
            final_seg_patch[:, dst_slices[0], dst_slices[1], dst_slices[2]] = seg_patch[:, src_slices[0], src_slices[1], src_slices[2]]
            
            # Store in the batch
            data[i] = final_data_patch
            seg[i] = final_seg_patch
            
            # Store regression value
            regression_values[i, 0] = regression_value
            
            # record properties of case for later reference
            properties_for_random_crop.append(case_props)
            selected_keys.append(key)
        
        # Create batch dict with all necessary information
        batch_dict = {
            'data': torch.from_numpy(data),
            'target': torch.from_numpy(seg),  # Use 'target' key for compatibility with original nnUNet
            'seg': torch.from_numpy(seg),     # Also provide 'seg' key for our custom code
            'properties': properties_for_random_crop,
            'keys': selected_keys,
            'regression_value': torch.from_numpy(regression_values)
        }
        
        return batch_dict
        
    def _apply_label_transform(self, seg: np.ndarray) -> np.ndarray:
        """
        Apply label transform to convert -1 values to 0 in segmentation
        
        Args:
            seg: Segmentation array
            
        Returns:
            Transformed segmentation array
        """
        # Create a temporary dictionary to match the expected format of RemoveLabelTansform
        temp_dict = {'data': None, 'seg': seg}
        
        # Apply the transform
        self.label_transform(**temp_dict)
        
        # Return the transformed segmentation
        return temp_dict['seg'] 