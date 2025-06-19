# _base_ = [
#     'mmdet::_base_/models/cascade-rcnn_hrnetv2p-w18.py',
#     'mmdet::_base_/datasets/coco_detection.py',
#     'mmdet::_base_/schedules/schedule_1x.py',
#     'mmdet::_base_/default_runtime.py'
# ]
_base_ = "mmdet::hrnet/cascade-rcnn_hrnetv2p-w18-20e_coco.py"


# ===================== MODEL =====================
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='Shared2FCBBoxHead', num_classes=63),
            dict(type='Shared2FCBBoxHead', num_classes=63),
            dict(type='Shared2FCBBoxHead',num_classes=63),
        ]
    )
)

# pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth"

# ===================== DATASET =====================
dataset_type = 'CocoDataset'
data_root = '/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0_masked_images_as_gt/'
classes = ['picea abies', 'pinus sylvestris', 'larix decidua',
       'fagus sylvatica', 'dead tree', 'abies alba',
       'pseudotsuga menziesii', 'acer pseudoplatanus',
       'fraxinus excelsior', 'acer sp.', 'tilia cordata', 'quercus sp.',
       'Tilia platyphyllos', 'Tilia spp', 'Ulmus glabra',
       'betula papyrifera', 'tsuga canadensis', 'acer saccharum',
       'betula sp.', 'picea rubens', 'betula alleghaniensis',
       'fagus grandifolia', 'picea sp.', 'acer pensylvanicum',
       'populus balsamifera', 'quercus ilex', 'quercus robur',
       'pinus strobus', 'larix laricina', 'larix gmelinii', 'pinus pinea',
       'populus grandidentata', 'pinus montezumae', 'betula pendula',
       'fraxinus nigra', 'dacrydium cupressinum', 'cedrus libani',
       'pinus elliottii', 'cryptomeria japonica', 'pinus koraiensis',
       'abies holophylla', 'alnus glutinosa', 'coniferous',
       'eucalyptus globulus', 'pinus nigra', 'quercus rubra',
       'tilia europaea', 'abies firma', 'metrosideros umbellata',
       'acer rubrum', 'picea mariana', 'abies balsamea',
       'castanea sativa', 'populus sp.', 'crataegus monogyna',
       'quercus petraea', 'acer platanoides', 'salix sp.', 'deciduous',
       'robinia pseudoacacia', 'pinus sp.', 'salix alba', 'carpinus sp.']
metainfo=dict(classes=classes)

img_scale = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
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
        metainfo=metainfo,
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

val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type

# ===================== OPTIMIZATION =====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
)

# ===================== LOGGING =====================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1
    )
]

visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer'
)
