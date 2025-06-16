_base_ = [
    'mmdet::_base_/models/cascade-rcnn_hrnetv2p-w18.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# ===================== MODEL =====================
model = dict(
    backbone=dict(frozen_stages=4),  # Freeze full HRNet backbone
    roi_head=dict(
        bbox_head=[
            dict(num_classes=53),
            dict(num_classes=53),
            dict(num_classes=53),
        ]
    )
)

# ===================== DATASET =====================
dataset_type = 'CocoDataset'
data_root = '/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/12_RGB_ObjDet_640_fL/'

img_scale = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/images/'),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# ===================== EVALUATION =====================
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator

# ===================== TRAINING =====================
train_cfg = dict(max_epochs=36, val_interval=1)

# ===================== OPTIMIZATION =====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# ===================== LOGGING =====================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
    param_scheduler=dict(type='MultiStepLR', milestones=[27, 33], gamma=0.1),
)

visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
