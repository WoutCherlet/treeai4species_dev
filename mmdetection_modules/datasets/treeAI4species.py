from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
import os

@DATASETS.register_module()
class TreeAI4SpeciesDataset(CocoDataset):
    def __init__(self, *args, mask_prefix=None, **kwargs):
        self.mask_prefix = mask_prefix
        self.pl_count = 0
        self.fl_count = 0
        super().__init__(*args, **kwargs)

    def parse_data_info(self, raw_data_info):
        data_info = super().parse_data_info(raw_data_info)

        # overwrite, if not present return none
        if self.mask_prefix is not None:
            filename = os.path.basename(data_info['img_path']) 
            mask_path = os.path.join(self.data_root, self.mask_prefix, filename)

            if os.path.exists(mask_path):
                self.pl_count += 1
                data_info['masked_img_path'] = mask_path
            else:
                # if no mask, assume fully labelled
                self.fl_count += 1
                data_info['masked_img_path'] = None

        return data_info
    
    def get_counts(self):
        # debug: check if loaded correctly
        return self.fl_count, self.pl_count