import os
import glob
import shutil
import json
import cv2

from mmengine.fileio import dump
from tqdm import tqdm
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


# TODO: change to process all folders at once

def data_to_coco_format(root_dir, output_dir):

    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # load classes from excel file
    # TODO: works now only for 12_RGB_...
    class_file = os.path.join(root_dir, "class12_RGB_all_L.xlsx")
    classes_df = pd.read_excel(class_file)
    classes = [{"id": int(row["Sp_ID"]), "name": row["Sp_Class"]} for _,row in classes_df.iterrows()]

    def process_split(split):
        image_id = 0
        ann_id = 0
        images = []
        annotations = []
        split_img_dir = os.path.join(root_dir, split, "images")
        split_lbl_dir = os.path.join(root_dir, split, "labels")
        output_img_dir = os.path.join(output_dir, split, "images")

        label_files = glob.glob(os.path.join(split_lbl_dir, "*.txt"))
        for label_path in tqdm(label_files, desc=f"Processing {split}"):
            fname = os.path.basename(label_path).replace(".txt", ".png")
            img_path = os.path.join(split_img_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: image not found or unreadable: {img_path}")
                continue
            height, width = img.shape[:2]

            # Copy image to output folder
            shutil.copy(img_path, os.path.join(output_img_dir, fname))

            # Add image entry
            images.append({
                "id": image_id,
                "file_name": fname,
                "width": width,
                "height": height
            })

            # Read label
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        print(f"Warning: malformed line: {line}, skipping.")
                        continue

                    class_id, x_c, y_c, w, h = map(float, parts)
                    class_id = int(class_id)
                    # Convert to absolute COCO format
                    abs_x = (x_c - w / 2) * width
                    abs_y = (y_c - h / 2) * height
                    abs_w = w * width
                    abs_h = h * height

                    annotation = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [abs_x, abs_y, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0,
                        "segmentation": []  # not available from bounding boxes
                    }
                    annotations.append(annotation)
                    ann_id += 1
            image_id += 1

        return images, annotations

    # Process both splits
    for split in ["train", "val"]:
        images, annotations = process_split(split)
        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": classes
        }
        ann_file = os.path.join(output_dir, "annotations", f"instances_{split}.json")
        dump(coco_dict, ann_file)
        print(f"Saved {split} annotations to {ann_file}")

    return

def get_list_of_classes():

    # harcoded
    # TODO: make this general for all files
    root_dir="/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL"
    class_file = os.path.join(root_dir, "class12_RGB_all_L.xlsx")

    classes_df = pd.read_excel(class_file)

    print(classes_df)

    classes_list = classes_df["Sp_Class"].to_numpy()

    print(repr(classes_list))
    print(len(classes_list))

    return classes_list





if __name__ == "__main__": 
    # Modify paths here
    # data_to_coco_format(
    #     root_dir="/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL",
    #     output_dir="/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/12_RGB_ObjDet_640_fL"
    # )

    get_list_of_classes()