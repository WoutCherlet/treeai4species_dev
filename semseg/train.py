import argparse
import yaml
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import albumentations as A

# Import own functionality
from utils import *


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    num_epochs,
    device,
    model_path,
    n_classes,
):
  
  # Model to device
  model = model.to(device)
  
  # Initialise validation iou
  iou_val_best = 0

  # Training loop
  for epoch in range(num_epochs):
    loss_train = []
    cm_train = np.zeros((n_classes, n_classes), dtype=np.int64)

    model.train()
    scaler = GradScaler(device)

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, masks in pbar:
        # Move tensors to the configured device (GPU or CPU)
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(images) # (batch_size, num_classes, height, width), logits
            loss, loss_dict = loss_fn(logits, masks)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Training metrics
        loss_train.append(loss.item())
        mask_pred = logits_to_labels(logits)
        mask_true = masks.cpu().numpy()
        cm_train += np.array(confusion_matrix(mask_true.flatten(), mask_pred.flatten(), labels=list(range(n_classes))))

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'CCE_tree': f"{loss_dict['tree_cce']:.4f}",
            'CCE': f"{loss_dict['cce']:.4f}",
            'Lovasz': f"{loss_dict['lovasz']:.4f}"
        })
    
    # Validation every x epochs
    val_interval = 1
    if epoch % val_interval == 0:
        cm_val = eval_confusion_matrix(val_loader, model, device, n_classes)
        iou_val = compute_iou(cm_val)
        iou_val_mean = np.mean(iou_val)
        if iou_val_mean > iou_val_best:
            model.save_pretrained(model_path)
            iou_val_best = iou_val_mean
    
    # Print metrics
    iou_train = np.mean(compute_iou(cm_train))
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(loss_train)}, IoU_train: {iou_train}, IoU_val: {iou_val_mean}")


def main(config):

    # Get configs
    n_classes = config.get("n_classes", 62)
    max_lr = config.get("max_lr", 1e-3)
    num_epochs = config.get("num_epochs", 100)
    batch_size = config.get("batch_size", 5)
    normalize = config.get("normalize", 'imagenet')
    loss_name = config.get("loss_name", 'cce')
    start_checkpoint = config.get("start_checkpoint", None)
    model_path = config.get("model_path", './model')
    model_name = config.get("model_name", 'segformer')
    data_paths = config.get("data_paths")

    # Set device
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    
    # Set normalization arguments
    if normalize['mode'] == 'custom':
        norm_kwargs = {
            'mean': normalize['mean'],
            'std': normalize['std'],
        }
    elif normalize['mode'] == 'imagenet':
        norm_kwargs = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
        }
    elif normalize['mode'] == 'image':
        norm_kwargs = {'normalization': 'image'}
    elif normalize['mode'] == 'image_per_channel':
        norm_kwargs = {'normalization': 'image_per_channel'}

    # Transforms
    train_transform = A.Compose([
        A.SquareSymmetry(p=1.0),
        A.Affine(
            scale=(0.95, 1.05),      # Zoom in/out by 80-120%
            rotate=(-5, 5),      # Rotate by -15 to +15 degrees
            translate_percent=(0, 0.05), # Optional: translate by 0-10%
            shear=(-5, 5),          # Optional: shear by -10 to +10 degrees
            p=1.0
        ),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(16, 32),
                        hole_width_range=(16, 32), fill=0, fill_mask=None, p=0.8),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MedianBlur(blur_limit=5, p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
        ], p=1.0),
        A.GaussNoise(std_range=(0.1, 0.2), p=0.3),
        A.Normalize(**norm_kwargs), #mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), normalization='image'
        A.ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Normalize(**norm_kwargs),
        A.ToTensorV2(),
    ]) 

    # Training dataset
    dataset = MyDataset(
        paths=[
            data_paths['dataset_1']['path_train'], 
            data_paths['dataset_2']['path_train'],
        ],
        labels=[
            'full',
            'partial',
        ],
        transform=train_transform,
    )

    # Validation set
    dataset_val = MyDataset(
        paths = [
            data_paths['dataset_1']['path_val'],
            data_paths['dataset_2']['path_val']],
        labels = [
            'full',
            'partial',
        ],
        transform=val_transform,
    )

    # Get class counts
    class_counts = get_class_count(dataset)
    class_counts = class_counts[1:]  # Discard the count for -1

    # Get sample counts
    sample_weights = get_sample_weights(dataset, n_classes)

    # Set training and validation loader
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    train_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    # Define model
    if start_checkpoint is not None:
        model = smp.from_pretrained(start_checkpoint)
        print(f'Start training from {start_checkpoint}.')
    elif model_name == 'segformer':
        model = smp.Segformer(
            encoder_name='mit_b5',      
            encoder_weights="imagenet",    
            in_channels=3,   
            classes=n_classes,                  
        )
    elif model_name == 'unet-resnet':
        model = smp.Unet(
            encoder_name='resnet34',      
            encoder_weights="imagenet",    
            in_channels=3,   
            classes=n_classes,                  
        )

    # Define loss function
    if loss_name == 'custom':
       loss_fn = CustomLoss(class_counts, device)
    elif loss_name == 'cce':
        class_frequency = class_counts / class_counts.sum()
        class_weights = torch.tensor(1 / np.log(class_frequency + 1.02)).to(torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)
    elif loss_name == 'lovasz':
       loss_fn = smp.losses.LovaszLoss('multiclass', per_image=False, ignore_index=-1, from_logits=True)

    # No weight decay on layernorm and bias
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if "bias" in name or "norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    # Set optimizer
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 5e-2}, # Set higher weight decay if overfitting
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=max_lr)

    # Set learning schedule
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
        anneal_strategy='cos'
    )

    # Train the model
    train_model(model, train_loader, val_loader, loss_fn, optimizer,
                scheduler, num_epochs, device, model_path, n_classes)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using a YAML config")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()

    config = parse_config(args.config)
    main(config)
