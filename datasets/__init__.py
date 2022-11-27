import torch
import torchvision
from .random_dataset import RandomDataset
from .liveness_dataset import *


def get_dataset_liveness(dataset, data_dir, transform, batch_size=None, labeled_ratio=0.8, ratio_live=0.0, ratio_spoof=0.0, fully_labeled=False, train=True, download=False, debug_subset_size=None, args=None):
    if dataset == 'ocim_ssdg_uniform':
        if train: # special
            assert(batch_size is not None)
            dataset = DatasetOCIM_SSDG_Uniform(data_dir, transform, train=train)
            sampler = BatchSampler_SSDG_Uniform(dataset, batch_size=batch_size, labeled_ratio=labeled_ratio)
            return dataset, sampler 
        else:
            dataset = DatasetOCIM_SSDG_Uniform(data_dir, transform, train=train)
    else:
        raise NotImplementedError
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch

    return dataset
