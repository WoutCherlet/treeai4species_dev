import os
import sys
from mmengine.registry import init_default_scope

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mmdetection_modules.datasets.treeAI4species import TreeAI4SpeciesDataset

init_default_scope('mmdet')

# Manually build the dataset
dataset = TreeAI4SpeciesDataset(
    ann_file='/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0/annotations/instances_train.json',
    data_root='/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0',
    data_prefix=dict(img='train/images/'),
    mask_prefix="train/masked_images/",
    filter_cfg=dict(filter_empty_gt=False),  # disable filtering for test
    pipeline=[],
    test_mode=False
)

fl_count, pl_count = dataset.get_counts()
print(f"fl count: {fl_count}")
print(f"pl count: {pl_count}")


print(f"Loaded {len(dataset)} items")
sample = dataset[0]

print("Sample keys:")
for k in sample:
    print(f"- {k}: {type(sample[k])}")

print(f"Image path: {sample['img_path']}")
print(f"Mask path: {sample['masked_img_path']}")

for i in range(2000, 2001):
    sample = dataset[i]
    
    print(f"Image path: {sample['img_path']}")
    print(f"Mask path: {sample['masked_img_path']}")
