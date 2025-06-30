_base_ = 'mmdet::swin/retinanet_swin-t-p4-w7_fpn_1x_coco.py'

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
albu_train_transforms = [
    dict(type="SquareSymmetry", p=1.0),
    dict(
        type='RandomBrightnessContrast',
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussianBlur', blur_limit=3, p=1),
            dict(type='MotionBlur', blur_limit=3, p=1)
        ],
        p=0.2),
    dict(
        type="GaussNoise",
        std_range=(0.1, 0.2),
        p=0.2
    )
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    # do cutout seperatly, cutout in albu creates errors (due to bboxs being filtered out and not being handled by mmdetection -____-)
    # dict(type="CutOut",
    #     cutout_ratio=[(0.05, 0.05), (0.01, 0.01), (0.03, 0.03)],
    #     n_holes=(0,3)),
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


# ===================== TRAINING =====================
train_cfg = dict(max_epochs=72, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05),
    accumulative_counts=2)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='CosineAnnealingLR',
        T_max=72,  # total epochs
        by_epoch=True,
        begin=0,
        end=72
    )
]

# ===================== LOGGING =====================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
)

# Optional: adjust working directory
work_dir = './work_dirs/retinanet_swin-t_treeai4species'