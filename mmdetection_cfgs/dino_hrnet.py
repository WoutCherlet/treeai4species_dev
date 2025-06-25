# config_dino_hrnet_tree.py

_base_ = 'mmdet::dino/dino-4scale_swin-l_12e_coco.py'  # base 12e COCO config

# --------------------- Data & Classes ---------------------
# Number of classes
num_classes = 63
class_names = [  # replace with your actual species list
    'species_1', 'species_2', ..., 'species_63'
]

# dataset settings
data_root = '/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/'
img_scale = (640, 640)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadLabelMask'),  # your mask loader
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        type='MaskedCocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/images/', mask='train/masks/'),
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    dataset=dict(
        type='MaskedCocoDataset',
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/', mask='val/masks/'),
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

# --------------------- Model ---------------------
model = dict(
    backbone=dict(
        type='HRNet',
        norm_eval=True,
        pretrained='open-mmlab://msra/hrnetv2_w18'
    ),
    bbox_head=dict(
        num_classes=num_classes,
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, class_weight=None),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=5.0)
    )
)

# load DINO pretrained weights (COCO): replace as needed
load_from = 'https://download.openmmlab.com/mmdetection/v3.3/dino/dino_4scale_r50_12e_coco_20220501/xxx.pth'

# --------------------- Optimizer / Scheduler ---------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, begin=0, by_epoch=False, end=1000),
    dict(type='MultiStepLR', begin=0, end=12, milestones=[8, 10], gamma=0.1, by_epoch=True)
]

# --------------------- Hooks ---------------------
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
)
