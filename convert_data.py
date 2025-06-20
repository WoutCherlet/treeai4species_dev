import os
import glob

import numpy as np

from PIL import Image

# TODO: also check if semantic segmentation data is the same pictures as object detection

# NOTES:

# 5_RGB_S_320_pL:
#   has labels 62,63,64 ??
# 0_rgb has different labels and winter images -> leave out for now

def get_all_unique_ids(label_folders):

    all_unique_ids = []

    for folder in label_folders:
        print(f"Processing folder {folder}")

        label_files = glob.glob(os.path.join(folder, "*.txt"))
        labels = []

        for label_file in label_files:
            bboxs = np.loadtxt(label_file)

            if len(bboxs.shape) == 1:
                # only one bounding box
                labels.append(bboxs[0])
            else:
                labels.extend(bboxs[:,0])

        labels = np.asarray(labels, dtype=int)
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        all_unique_ids.extend(unique_labels)

        print(len(unique_labels))
        print(unique_labels)
        print(label_counts)

    unique_final = np.unique(all_unique_ids)

    print("Final unique labels")
    print(unique_final)
    print(len(unique_final))


def convert_0_rgb_labels(label_folder_0_rgb, out_folder):
    # TODO: convert the labels of 0_rgb (see readme for conversion)

    # for now, exclude this set
    # also issues with dead trees being labelled, maybe easier to exclude alltogether
    return

def check_masked_image(image_path, masked_image_path):
    # check out format of masked image (just black color?)

    im = Image.open(image_path)
    masked_im = Image.open(masked_image_path)

    imarray = np.array(im)
    maskedimarray = np.array(masked_im)

    print(imarray.shape)
    print(maskedimarray.shape)

    print(imarray)
    print(maskedimarray)

    return


# label_folder_5_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/5_RGB_S_320_pL/train/labels/"
# label_folder_0_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/0_RGB_FullyLabeled/coco/train/labels/"
# label_folder_12_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL/train/labels/"
# label_folder_34a_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_a/34_RGB_ObjDet_640_pL/train/labels/"
# label_folder_34b_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_b/train/labels/"

label_folder_5_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/5_RGB_S_320_pL/val/labels/"
label_folder_0_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/0_RGB_FullyLabeled/coco/val/labels/"
label_folder_12_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL/val/labels/"
label_folder_34a_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_a/34_RGB_ObjDet_640_pL/val/labels/"
label_folder_34b_rgb = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_b/val/labels/"


label_folders = [label_folder_5_rgb, label_folder_0_rgb, label_folder_12_rgb, label_folder_34a_rgb, label_folder_34b_rgb]

get_all_unique_ids(label_folders)

