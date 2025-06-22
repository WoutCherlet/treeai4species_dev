from mmdet.registry import DATA_SAMPLERS
from torch.utils.data import WeightedRandomSampler

import numpy as np

@DATA_SAMPLERS.register_module()
class WeightedRandomSamplerMod(WeightedRandomSampler):
    def __init__(self, dataset, class_weights, replacement=True, seed= None):
        class_weights = np.asarray(class_weights)
        sample_weights = []

        self.dataset = dataset
        self.epoch = 0
        
        for idx in range(len(dataset)):
            data_info = dataset.get_data_info(idx)

            all_labels_img = []
            for annotation in data_info["instances"]:
                label = annotation['bbox_label']
                all_labels_img.append(label)

            if len(all_labels_img) == 0:
                sample_weight = 1
            else:
                unique_labels = np.unique(all_labels_img)
                sample_weight = class_weights[unique_labels].sum()

            sample_weights.append(sample_weight)
        
        super().__init__(sample_weights, len(dataset), replacement=replacement)