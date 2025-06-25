_base_ = 'mmdetection/configs/swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py'

# Update number of classes
model = dict(
    bbox_head=dict(
        num_classes=63 
    )
)

# Define dataset paths
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
    # cutout should help a lot with occluded trees
    dict(
        type='CutOut',
        n_holes = (0,3),
        cutout_ratio=[(0.05, 0.05), (0.03, 0.03), (0.07, 0.07)]
    ),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='train/images/'),
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

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# try 40 epochs
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[25, 33],
        gamma=0.1
    )
]

# ===================== LOGGING =====================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
)

# Optional: adjust working directory
work_dir = './work_dirs/retinanet_swin-t_treeai4species'