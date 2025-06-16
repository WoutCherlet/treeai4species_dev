import os

import pandas as pd


def label_file_to_list(label_file):
    # # format: class x_center y_center width height
    data = pd.read_csv(label_file, sep=" ", header=None)
    data.columns = ["label", "x_center", "y_center", "width", "height"]
    return data

def xy_to_img_coordinates(labels, img):
    height, width, _ = img.shape
    labels["x_center"] = labels["x_center"]*width
    labels["width"] = labels["width"]*width
    labels["y_center"] = labels["y_center"]*height
    labels["height"] = labels["height"]*height
    return labels


def label_files_to_coco(label_dir):

    # todo: convert label files to coco json file (see openmm website)

    raise NotImplementedError