import os
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS
from PIL import Image


@TRANSFORMS.register_module()
class LoadLabelMask(BaseTransform):
    """Load a labeled area mask (black = unlabeled, color = labeled)."""

    def transform(self, results):
        mask_path = results.get('mask_path', None)

        if mask_path is None or not os.path.exists(mask_path):
            # No mask: assume everything is labeled
            results['label_mask'] = np.ones((results['img_shape'][0], results['img_shape'][1]), dtype=bool)
            return results

        # Load mask as RGB image
        mask_img = Image.open(mask_path).convert('RGB')
        mask = np.array(mask_img)

        # Everything not (0,0,0) is labeled
        label_mask = ~np.all(mask == 0, axis=-1)  # shape: (H, W), dtype=bool
        results['label_mask'] = label_mask
        return results