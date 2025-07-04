import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import albumentations as A

# Import own functionality
from utils import *


def evaluate(dataloader, model, device, n_classes):
    # Get confusion matrix
    cm = eval_confusion_matrix(dataloader, model, device, n_classes)
    iou_per_class = compute_iou(cm)
    iou_mean = np.mean(iou_per_class)
    print('mean IoU:', f"{iou_mean * 100:.1f}%")
    for i, iou in enumerate(iou_per_class):
        print(f"Class {i}: IoU = {iou * 100:.1f}%")


def plot_predictions(dataset, model, device, n=5):
    idx = np.random.choice(len(dataset), size=n, replace=False)
    for id in idx:
        img, mask = dataset[id]
        mask_pred = predict_patch(img, model, device)
        plot_prediction_vs_truth(img, mask, mask_pred)


def main(config):

    # Get configs
    n_classes = config.get("n_classes", 62)
    batch_size = config.get("batch_size", 5)
    model_path = config.get("model_path", './model')
    data_paths = config.get("data_paths")
    normalize = config.get("normalize", 'imagenet')

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

    # Transform
    transform = A.Compose([
        A.Normalize(**norm_kwargs), # mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ]) 

    # Training dataset
    print('Loading training set ...')
    dataset = MyDataset(
        paths=[
            data_paths['dataset_1']['path_train'], 
            data_paths['dataset_2']['path_train']
        ],
        labels=[
            'full',
            'partial',
        ],
        transform=transform
    )

    print('Loading validation set ...')
    # Validation set
    dataset_val = MyDataset(
        paths = [
            data_paths['dataset_1']['path_val'],
            data_paths['dataset_2']['path_val']],
        labels = [
            'full',
            'partial',
        ],
        transform=transform,
    )

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

    # Get model
    print(f'Loading model {model_path}')
    model = smp.from_pretrained(model_path)

    # Calculate IoU
    print('Evaluating on training set')
    evaluate(train_loader, model, device, n_classes)

    print('Evaluating on validation set')
    evaluate(val_loader, model, device, n_classes)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model, given a YAML config")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()

    config = parse_config(args.config)
    main(config)