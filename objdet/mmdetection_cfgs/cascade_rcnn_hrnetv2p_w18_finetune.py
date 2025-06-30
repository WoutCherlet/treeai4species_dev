_base_ = "mmdet::hrnet/cascade-rcnn_hrnetv2p-w18-20e_coco.py"

# ADAPT: path of pretrained from first iteration
load_from = "/home/wcherlet/TreeAI4Species/treeai4species_dev/work_dirs/cascade_rcnn_hrnetv2p_w18/20250619_170024/best_coco_bbox_mAP_50_epoch_35.pth"

# ===================== DATASET =====================
dataset_type = 'CocoDataset'
# ADAPT: data root (as used in data preprocessing notebook)
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

# custom normalization
data_mean_normalize_full = [72.16316447706907, 94.31619658327361, 79.79854683068336]
data_std_normalize_full = [50.2286901789422, 53.97488884710553, 47.07236402248264]

metainfo=dict(classes=classes)
img_scale = (640, 640)

# albumentation augmentations
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
    dict(type="CutOut",
        cutout_ratio=[(0.05, 0.05), (0.01, 0.01), (0.03, 0.03)],
        n_holes=(0,3)),
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

# ===================== MODEL =====================
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=data_mean_normalize_full,
        std=data_std_normalize_full,
        bgr_to_rgb=True,
        pad_size_divisor=32),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead', 
                num_classes=63,
            ),
            dict(
                type='Shared2FCBBoxHead', 
                num_classes=63,
            ),
            dict(
                type='Shared2FCBBoxHead', 
                num_classes=63,
            ),
        ]
    )
)


# ===================== EVALUATION =====================
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator

# ===================== TRAINING =====================
# for finetuning with load_from: extra number of epochs here
train_cfg = dict(max_epochs=30, val_interval=1)

val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type

# ===================== OPTIMIZATION =====================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
)

# ===================== LOGGING =====================
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', rule='greater'),
)

# finetuning scheduler
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            15,
            25,
        ],
        type='MultiStepLR'),
]

visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')],
    name='visualizer'
)



