import matplotlib.pyplot as plt
import numpy as np

from mmengine.registry import init_default_scope
from mmdet.registry import TRANSFORMS

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmdetection_modules.datasets.transforms.load_label_mask import LoadLabelMask

image_path = '/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0/train/images/5_RGB_S_320_pL_train_2017_000000000000.tif'
mask_path = '/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0/train/masked_images/5_RGB_S_320_pL_train_2017_000000000000.tif' 

# mock pipeline input
img = plt.imread(image_path)
H, W = img.shape[:2]
results = {
    'img_path': image_path,
    'img': img,
    'img_shape': (H, W),
    'ori_shape': (H, W),
    'mask_path': mask_path
}
# results = {
#     'img_path': image_path,
#     'img': img,
#     'img_shape': (H, W),
#     'ori_shape': (H, W),
#     'mask_path': None
# }

init_default_scope('mmdet')

transform = TRANSFORMS.get('LoadLabelMask')()
transformed = transform(results)

# 3. Visualize
label_mask = transformed['label_mask']

print(label_mask)
print(np.unique(label_mask))

# plot mask
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Input Image')
plt.imshow(img)

masked_img = plt.imread(mask_path)
plt.subplot(1, 3, 2)
plt.title('Masked input Image')
plt.imshow(masked_img)

plt.subplot(1, 3, 3)
plt.title('Labeled Area Mask')
plt.imshow(label_mask, cmap='gray')
plt.tight_layout()
plt.savefig("tests/test.png")