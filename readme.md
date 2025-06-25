# TreeAI4Species: object detection and semantic segmentation models

## TODOS
- notebook preprocessing
- refs in description of model
- different config files for original and 
- dependencies file for albumentations, pillow etc.
- adapt text to reflect final training and models

Code and documentation for TreeAI4Species submissions, for both the object detection and semantic segmentation tracks.  
Instructions for installation, preprocessing, training and a description of model are described below for both tracks

# Object detection

# Installation

We use mmdetection (v3.3.0) as the framework for object detection.
We used python v3.9, pytorch v2.5.1 with cuda v12.4.

To install, follow the instructions [here](https://mmdetection.readthedocs.io/en/latest/get_started.html).


## Data preprocessing

Due to time constraints and the need for manual filtering, as well as the 3 classes already being well represented, the 0_RGB_fL data subset was not used.

The other first subset are merged, converted to coco format and a final class list and mapping are extracted.
Mean RGB and std values are extracted from the dataset.
(Some of the class information has been manually copied to the configuration file)

## Training model

The model is trained using:

python train.py mmdetection_cfgs/cascade_rcnn_hrnetv2p_w18.py

TODO finetuning run here, get different config so command is easy


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
A second finetuning run was performed, with the above augmentations and normalization, and with a restarted scheduler to simulate a warm restart.


