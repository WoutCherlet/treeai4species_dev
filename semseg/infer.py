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

# Import own functionality
from utils import *


def main(config):

    # Get configs
    model_path = config.get("model_path", './model')
    data_paths = config.get("data_paths")

    # Directories 
    dir_test = data_paths['data_test']['path_test']
    dir_predictions = data_paths['data_test']['path_prediction']

    # Set device
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    # Test dataset
    print('Loading test set ...')
    dataset = MyDataset(dir_test, mode='infer')

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