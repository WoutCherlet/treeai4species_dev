_base_ = "mmdet::hrnet/cascade-rcnn_hrnetv2p-w18-20e_coco.py"

# pretrained weights
load_from = "https://download.openmmlab.com/mmdetection/v2.0/hrnet/cascade_rcnn_hrnetv2p_w18_20e_coco/cascade_rcnn_hrnetv2p_w18_20e_coco_20200210-434be9d7.pth"

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
class_frequencies = [8.90876430e-02, 5.31485925e-02, 5.09251699e-02, 3.22961549e-02,
       1.52687578e-01, 2.14177145e-02, 1.03634103e-02, 5.89144171e-03,
       5.46434359e-03, 1.04136571e-02, 8.66757948e-04, 1.00116824e-02,
       1.25617094e-05, 3.41678496e-03, 2.51234188e-05, 1.39045561e-01,
       1.29373045e-01, 5.98565453e-02, 3.00978557e-02, 2.52992827e-02,
       1.84029043e-02, 1.73225973e-02, 1.66191415e-02, 1.09035638e-02,
       9.91118871e-03, 6.19292273e-03, 9.92375042e-03, 2.60027385e-03,
       6.94662530e-03, 8.49171555e-03, 3.51727863e-03, 3.73082769e-03,
       3.10274222e-03, 5.57739897e-03, 2.82638461e-03, 3.06505709e-03,
       3.12786564e-03, 2.36160137e-03, 2.96456342e-03, 1.78376273e-03,
       1.63302222e-03, 2.62539726e-03, 1.97218838e-03, 1.74607761e-03,
       1.62046051e-03, 2.75101436e-03, 7.91387692e-04, 5.90400342e-04,
       6.40647179e-04, 9.29566495e-04, 8.29072820e-04, 8.79319658e-04,
       8.29072820e-04, 4.39659829e-03, 1.49484342e-03, 1.33154120e-03,
       1.10543043e-03, 1.36922632e-03, 2.22342256e-03, 3.76851282e-04,
       4.14536410e-04, 2.13549060e-04, 1.63302222e-04]
# calculated as 1/np.log(1.04 + freq)
class_weights = [ 8.23655986, 11.2281144 , 11.49080353, 14.32617788,  5.67507163,
       16.77698062, 20.35159027, 22.28681633, 22.49155375, 20.33179632,
       24.96642817, 20.49126152, 25.48888208, 23.52902519, 25.4810374 ,
        6.07145134,  6.39109353, 10.50643563, 14.7601262 , 15.80883294,
       17.61770632, 17.94048243, 18.15728218, 20.14085261, 20.53152909,
       22.14457488, 20.52648678, 23.97055112, 21.79696257, 21.11813332,
       23.47582847, 23.36359711, 23.6968635 , 22.43698519, 23.84659706,
       23.71716857, 23.6833465 , 24.10282686, 23.77148953, 24.42933058,
       24.51599581, 23.95671352, 24.32187431, 24.45093822, 24.52324625,
       23.88776941, 25.011647  , 25.13305141, 25.10258775, 24.92887302,
       24.98901672, 24.95890791, 24.98901672, 23.02062839, 24.59599202,
       24.69122249, 24.82432906, 24.66917951, 24.18009096, 25.26336775,
       25.24027069, 25.36395519, 25.39506971]
# extra weight for background class (seems to be necessary)
# set to lowest value of class_weights
# class_weights += [1.0]
# normalize class weights (also seems to be necessary)
s = sum(class_weights)
class_weights = [w/s for w in class_weights]

# custom normalization
data_mean_normalize_full = [72.16316447706907, 94.31619658327361, 79.79854683068336]
data_std_normalize_full = [50.2286901789422, 53.97488884710553, 47.07236402248264]

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
    # weighted random sampler (custom module), image are weighted by sum of class weights of unique labels in image
    # sampler=dict(
    #     _delete_=True,
    #     type='WeightedRandomSamplerMod',
    #     class_weights=class_weights,
    #     replacement=True
    # ),
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
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     class_weight=class_weights,
                #     loss_weight=1.0
                # )
            ),
            dict(
                type='Shared2FCBBoxHead', 
                num_classes=63,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     class_weight=class_weights,
                #     loss_weight=1.0
                # )
            ),
            dict(
                type='Shared2FCBBoxHead', 
                num_classes=63,
                # loss_cls=dict(
                #     type='CrossEntropyLoss',
                #     use_sigmoid=False,
                #     class_weight=class_weights,
                #     loss_weight=1.0
                # )
            ),
        ]
    )
)


# ===================== EVALUATION =====================
val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/instances_val.json', metric='bbox', format_only=False)
test_evaluator = val_evaluator

# ===================== TRAINING =====================
# train_cfg = dict(max_epochs=50, val_interval=1)
# for finetuning with load_from: extra number of epochs here
train_cfg = dict(max_epochs=36, val_interval=1)

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



