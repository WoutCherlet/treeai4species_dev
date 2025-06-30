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


def main(config):

    # Get configs
    model_path = config.get("model_path", './model')
    data_paths = config.get("data_paths")
    normalize = config.get("normalize", 'imagenet')

    # Directories 
    dir_test = data_paths['data_test']['path_test']
    dir_predictions = data_paths['data_test']['path_prediction']
    os.makedirs(dir_predictions, exist_ok=True)

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

    # Transforms
    transform = A.Compose([
        A.Normalize(**norm_kwargs), 
        A.ToTensorV2(),
    ]) 

    # Test dataset
    print('Loading test set ...')
    dataset = MyDataset(dir_test, mode='infer', transform=transform)

    # Get model
    print(f'Loading model {model_path}')
    model = smp.from_pretrained(model_path)

    for i in tqdm(range(len(dataset)), desc='Inference'):
        # Get test image
        img = dataset[i]
        filename = dataset.names[i]

        # Make prediction
        mask = predict_patch(img, model, device)

        # Save prediction
        name_out = dir_predictions + filename[:-3] + 'npy'
        np.save(name_out, mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference, given a YAML config")
    parser.add_argument("config", type=str, help="Path to config.yml")
    args = parser.parse_args()

    config = parse_config(args.config)
    main(config)