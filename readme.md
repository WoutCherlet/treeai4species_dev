# TreeAI4Species: object detection and semantic segmentation models

Code and documentation for TreeAI4Species submissions, for both the object detection and semantic segmentation tracks.  
Instructions for installation, preprocessing, training and a description of model are described below for both tracks.

Contact: wout.cherlet@ugent.be, wouter.vandenbroeck@ugent.be

# Object detection

# Installation

We use mmdetection (v3.3.0) as the framework for object detection.
We used python v3.10, pytorch v2.5.1 with cuda v12.4.

To install mmdetection, follow the instructions [here](https://mmdetection.readthedocs.io/en/latest/get_started.html). Make sure to install from source.
> Note: mmcv must be [installed from source](https://mmcv.readthedocs.io/en/latest/get_started/build.html), instead of using mim.
> Note: you may need to adapt line 9 in mmdetection/mmdet/__init__.py to 
```python
mmcv_maximum_version = '2.2.1'
```

An environment.yml file is available for all other dependencies.

## Data preprocessing

Due to time constraints and the need for manual filtering, as well as the 3 classes already being well represented, the 0_RGB_fL data subset was not used. The other four subset are merged, converted to coco format and a final class list and mapping are extracted.  
Mean RGB and std values were extracted from the dataset.
A notebook (prepare_data.ipynb) is provided to convert the data to a single coco dataset.

## Training model

> Note: paths to dataset root and checkpoints must be adapted in the config files.

The model is trained using:
```
python train.py mmdetection_cfgs/cascade_rcnn_hrnetv2p_w18.py
```
A second config file is provided for the finetuning run:
```
python train.py mmdetection_cfgs/cascade_rcnn_hrnetv2p_w18_finetune.py
```

## Inference

Run inference with the following command:
```
python inference.py --config [config_path] --checkpoint [ckpt_path] --test_dir [dir with test images] --output_dir [output dir] --conf_threshold [conf threshold]
```
Afterwards, convert the data to submission format and labels (mmdetection internal labels are different!). A notebook (postprocess_data.ipynb) is provided for this.


## Description of model

The model consists of a standard HRNet backbone [(ref)](https://doi.org/10.48550/arXiv.1908.07919) with a three-stage Cascade R-CNN prediction model [(ref)](https://doi.org/10.48550/arXiv.1712.00726).
A standard pretrained ImageNet checkpoint is used as initial weights. No changes to these architectures were made.

Data was augmented as follows (default parameters unless mentioned):
- SquareSymmetry (albumentations), p=0.1
- RandomBrightnessContrast (albumentations), p=0.2
- One of GaussianBlur or MotionBlur (albumentations), both blur_limit=3, p=0.2
- Gaussian Noise (albumentations), std_range=(0.1, 0.2), p=0.2
- CutOut (mmdetection), cutout_ratio=[(0.05, 0.05), (0.01, 0.01), (0.03, 0.03)], n_holes=(0,3)

The model was trained without augmentations or normalization for 35 epochs, leading to an mAP@0.50 of 0.48.
A second finetuning run was performed for 32 epochs, with the above augmentations and normalization, and with a restarted scheduler to simulate a warm restart.

Final model predictions are post-processed by first filtering out duplicate detections, filtering out bounding boxes with IoU overlap of >0.85 and keeping the box and class with highest confidence score. Finally, a confidence threshold of 0.2 was used.


# Semantic Segmentation

All functionality for semantic segmenation is grouped into the `semseg` directory.

**Installation**: make new conda environment (see `environment.yml`)

**Key ideas of the approach**:
- Data: both fully labeled and partially labeled
- Model: SegFormer [ref](https://doi.org/10.48550/arXiv.2105.15203) with mitb5 backbone
- Loss function: custom combination of Lovasz and categorical cross entropy (CCE) for both per species and background vs tree. For the partially labeled data, the loss is only calculated where there is a label
- Class weighting: both pixel-wise class weights for CCE and image-wise sample weights used in the data generator (WeightedRandomSampler). Weights are calculated as $1 / log(1.2 + f_c)$ wit $f_c$ the class frequency. For the sample weights, the image-wise class weights are summed up if they occur for each image.
- Transforms: normalization with mean and std calculated from dataset, basic geometric and photometric transforms, blur and noise, coarse dropout
- Optimization: AdamW (weight decay not for bias and norm layers) and OneCycleLR schedule
- Training: start from imagenet weights, evaluate on validation set each epoch and maintain only model with highest mIoU_val

\
All data paths and high level hyperparameters are specified in a `config.yml` file.

**Training:**

```
python train.py config.yml
```

**Inference:**

```
python infer.py config.yml
```
