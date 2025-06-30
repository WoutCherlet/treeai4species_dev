# TreeAI4Species: object detection and semantic segmentation models

## TODOS
- notebook preprocessing
- refs in description of model
- different config files for original and 
- dependencies file for albumentations, pillow etc.
- adapt text to reflect final training and models
- add trained checkpoints in 

Code and documentation for TreeAI4Species submissions, for both the object detection and semantic segmentation tracks.  
Instructions for installation, preprocessing, training and a description of model are described below for both tracks

# Object detection

# Installation

We use mmdetection (v3.3.0) as the framework for object detection.
We used python v3.9, pytorch v2.5.1 with cuda v12.4.

To install, follow the instructions [here](https://mmdetection.readthedocs.io/en/latest/get_started.html).  
> Note: you may need to adapt line 9 in mmdetection/mmdet/__init__.py to 
```python
mmcv_maximum_version = '2.2.1'
```

## Data preprocessing

Due to time constraints and the need for manual filtering, as well as the 3 classes already being well represented, the 0_RGB_fL data subset was not used. The other four subset are merged, converted to coco format and a final class list and mapping are extracted.  
Mean RGB and std values were extracted from the dataset.

## Training model

The model is trained using:
```
python train.py mmdetection_cfgs/cascade_rcnn_hrnetv2p_w18.py
```
TODO finetuning run here, get different config so command is easy

## Inference

Run inference with the following command:
```
python inference.py --config [config_path] --checkpoint [ckpt_path] --test_dir [dir with test images] --output_dir [output dir] --conf_threshold [conf threshold]
```
Afterwards, convert the data to submission format and labels (mmdetection internal labels are different!):

TODO script for this


## Description of model

The model consists of a standard HRNet backbone (ref) with a three-stage Cascade R-CNN prediction model (ref).
A standard pretrained ImageNet checkpoint is used as initial weights. No further changes to the architecture were made.

Data was augmented as follows (default parameters unless mentioned):
- SquareSymmetry (albumentations), p=0.1
- RandomBrightnessContrast (albumentations), p=0.2
- One of GaussianBlur or MotionBlur (albumentations), both blur_limit=3, p=0.2
- Gaussian Noise (albumentations), std_range=(0.1, 0.2), p=0.2
- CutOut (mmdetection), cutout_ratio=[(0.05, 0.05), (0.01, 0.01), (0.03, 0.03)], n_holes=(0,3)

The model was trained without augmentations or normalization for 35 epochs, leading to an mAP@0.50 of 0.48.
A second finetuning run was performed for 32 epochs, with the above augmentations and normalization, and with a restarted scheduler to simulate a warm restart.

Final model predictions are post-processed by first filtering out duplicate detections, filtering out bounding boxes with IoU overlap of >0.85 and keeping the box and class with highest confidence score. Finally, a confidence threshold of 0.2 was used.


