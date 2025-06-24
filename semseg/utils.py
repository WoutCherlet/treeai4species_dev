import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
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


def parse_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class MyDataset(Dataset):
  def __init__(self, paths, labels='full', mode='train', transform=None):
    self.path = [paths] if not isinstance(paths, list) else paths
    self.labels = [labels] if not isinstance(labels, list) else labels
    self.mode = mode
    self.transform = transform
    self.images, self.masks = self._read_dataset() 

  def __len__(self):
    return len(self.images)

  def _read_dataset(self):
    images, masks = [], []
    for path, labels in zip(self.path, self.labels):
      self.names = [f for f in os.listdir(path + 'images/') if f[-3:] == 'png']
      for name in self.names:
          image_path = path + 'images/' + name
          try:
            with Image.open(image_path) as image:
              images.append(np.array(image))
            if self.mode != 'infer': 
                mask_path = path + 'labels/' + name
                with Image.open(mask_path) as m:
                  mask = np.array(m).astype(np.int64)
                  if labels == 'partial':
                    mask[mask == 0] = -1
                  masks.append(mask)
          except Exception as e:
              print(f"Warning: Could not read file '{name}': {e}")
    
    return images, masks

  def __getitem__(self, idx):
    image = self.images[idx]
    # image = np.moveaxis(image, -1, 0).astype(np.float32) / 255

    if self.mode == 'infer':
      image = self.transform(image=image)['image'] if self.transform else image
      return image
    else:
      mask = self.masks[idx]
      if self.transform:
          augmented = self.transform(image=image, mask=mask)
          image = augmented['image']
          mask = augmented['mask'].long()
      return image, mask


def get_class_count(dataset, plot=False):
    masks_all = np.stack(dataset.masks, axis=0)
    # Get unique values and their counts
    unique_vals, counts = np.unique(masks_all.flatten(), return_counts=True)
    
    if plot:
        # Print results
        for val, count in zip(unique_vals, counts):
            print(f"Value: {val}, Count: {count}")

        # Plot a bar chart
        plt.figure(figsize=(10, 5))
        plt.bar(unique_vals, counts, color='skyblue')
        plt.xlabel("Class ID")
        plt.ylabel("Count (log scale)")
        plt.yscale('log')
        # plt.title("Histogram of Unique Pixel Values")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return counts


def get_sample_weights(dataset, n_classes):
    masks_all = np.stack(dataset.masks, axis=0) # N, H, W
    n = masks_all.shape[0]
    masks_all = masks_all.reshape(n, -1)
    
    class_counts = np.zeros(n_classes, dtype=np.uint64)
    for mask in masks_all:
        u = np.unique(mask)
        u = u[u != -1]
        class_counts[u] += 1

    class_frequencies = class_counts / n
    class_weights = 1 / np.log(1.02 + class_frequencies)

    sample_weights = np.zeros(n, dtype=np.int64)
    for i, mask in enumerate(masks_all):
        # Get unique ID's (ID = index)
        u = np.unique(mask)
        sample_weights[i] = class_weights[u].sum()
    
    return sample_weights


def logits_to_labels(logits):
    # Take the argmax over the num_classes dimension (axis=1)
    labels = torch.argmax(logits, dim=1)
    # Convert to NumPy array
    labels = labels.cpu().numpy().astype(np.int64)
    return labels


def eval_confusion_matrix(dataloader, model, device, n_classes=62):
    model.eval()
    model.to(device)
    # During inference, disable gradient computation using `torch.no_grad()`
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            logits = model(images)
            mask_pred = logits_to_labels(logits)
            cm += np.array(confusion_matrix(masks.flatten(), mask_pred.flatten(), labels=list(range(n_classes))))
    return cm


def predict_patch(img_patch, model, device):
  # Set model in evaluation mode (Dropout is disabled, and BatchNorm uses running averages instead of batch statistics)
  model.eval()
  model.to(device)
  # During inference, disable gradient computation using `torch.no_grad()`
  with torch.no_grad():
    # Process image into correct model input format
    input = torch.tensor(img_patch).unsqueeze(0).to(device) # (C, H, W) to (1, C, H, W)
    # Make model prediction
    logits = model(input)
  # Process model output
  mask = logits_to_labels(logits)[0] # (H, W)
  return mask


def plot_prediction_vs_truth(img, mask_true, mask_pred):
  # Create a figure with 3 subplots
  fig, axes = plt.subplots(1, 3, figsize=(12, 4))
  # Plot original image (RGB)
  axes[0].imshow(np.moveaxis(img, 0, -1))
  axes[0].set_title('Input Image')
  # Plot mask prediction
  axes[1].imshow(mask_pred)
  axes[1].set_title('Mask Prediction')
  # Plot ground truth mask
#   cmap = ListedColormap(['purple', 'black', 'white'])
  # bounds = [-1.5, -0.5, 0.5, 1.5]
#   norm = plt.Normalize(vmin=-1.5, vmax=1.5)
  axes[2].imshow(mask_true)
  axes[2].set_title('Ground Truth Mask')
  # Show the plot
  plt.tight_layout()
  plt.show()


def compute_iou(conf_matrix):
    # True Positives: diagonal elements
    TP = np.diag(conf_matrix)
    # False Positives: sum over column - TP
    FP = np.sum(conf_matrix, axis=0) - TP
    # False Negatives: sum over row - TP
    FN = np.sum(conf_matrix, axis=1) - TP
    # Union = TP + FP + FN
    union = TP + FP + FN
    # Calculate IoU
    iou = TP / (union + 0.00001)
    return iou


def plot_confusion_matrix(conf_matrix, class_names):
    df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


class CustomLoss:
    def __init__(self, class_counts, device):
        # Weighted CCE loss for all classes
        class_frequency = class_counts / class_counts.sum()
        class_weights = torch.tensor(1 / np.log(class_frequency + 1.02)).to(torch.float32).to(device)
        self.cce = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights)

        # Binary CE for tree vs no tree
        class_count_tree_notree = np.array([class_counts[0], class_counts[1:].sum()])
        class_freq_tree_notree = class_count_tree_notree / class_count_tree_notree.sum()
        class_weight_tree_notree = torch.tensor(1 / np.log(class_freq_tree_notree + 1.02)).to(torch.float32).to(device)
        self.cce_tree = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weight_tree_notree)

        # Lovasz loss
        self.lovasz = smp.losses.LovaszLoss('multiclass', per_image=False, ignore_index=-1, from_logits=True)

    def __call__(self, logits, target):
        # Weighted CCE loss for tree vs no-tree
        logits_tree = torch.cat([
            logits[:, 0:1, :, :],
            logits[:, 1:, :, :].sum(dim=1, keepdim=True)
        ], dim=1)
        target_tree = torch.where(target > 0, 1, target)
        cce_tree_loss = self.cce_tree(logits_tree, target_tree)

        # Weighted CCE loss for all classes -> per pixel
        cce_loss = self.cce(logits, target)

        # Lovasz loss for each class
        lovasz_loss = self.lovasz(logits, target)

        # Combined
        total_loss = (
            0.3 * cce_loss +  
            0.1 * cce_tree_loss +  
            1 * lovasz_loss  
        )

        loss_dict = {
            'lovasz': lovasz_loss.item(),
            'cce': cce_loss.item(),
            'tree_cce': cce_tree_loss.item(),
        }

        return total_loss, loss_dict