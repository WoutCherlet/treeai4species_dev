import os
import glob
import shutil
import json
import cv2

from mmengine.fileio import dump
from tqdm import tqdm
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import patches

def label_file_to_list(label_file, submission_output=False):
    if submission_output:
        # format: img_id species x_center y_center width height conf
        data = pd.read_csv(label_file, 
                           sep=" ", 
                           header=None, 
                           names=["img_id", "species", "x_center", "y_center", "width", "height", "conf"],
                           dtype={"img_id": str})
    else:
        # input labels
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

def plot_image_with_boxes(image_file, label_file):
    # read image and bboxs
    bboxs = label_file_to_list(label_file)
    img = cv2.imread(image_file)

    # convert bboxs to image coordinates
    bboxs = xy_to_img_coordinates(bboxs, img)

    # plot
    fig, ax = plt.subplots()
    ax.imshow(img)
    for _, row in bboxs.iterrows():
        x = row["x_center"] - row["width"]/2
        y = row["y_center"] - row["height"]/2
        w = row["width"]
        h = row["height"]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x,y, int(row["label"]), fontsize=8, color="white", bbox=dict(facecolor='red', edgecolor="none"))
        
    plt.show()
    return 

# for all datasets
def data_to_coco_format(root_dirs, class_dict, output_dir, masked_as_gt=False):
    """
    Convert data from multiple root directories to COCO format.
    
    Args:
        root_directories: List of root directory paths, each containing train/val folders
        class_dict: Dictionary mapping class IDs to class names {id: name}
        output_dir: Output directory for the combined COCO dataset
        masked_as_gt: if true, writes masked image as the gt image
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    
    # Convert class dictionary to COCO categories format
    categories = [{"id": int(class_id), "name": class_name} for class_id, class_name in class_dict.items()]
    
    def process_split(split):
        image_id = 0
        ann_id = 0
        images = []
        annotations = []
        
        output_img_dir = os.path.join(output_dir, split, "images")
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)
        if not masked_as_gt:
            output_masked_img_dir = os.path.join(output_dir, split, "masked_images")
            if not os.path.exists(output_masked_img_dir):
                os.makedirs(output_masked_img_dir)
        
        # Process each root directory
        for root_dir in root_dirs:
            split_img_dir = os.path.join(root_dir, split, "images")

            split_img_masked = os.path.join(root_dir, split, "images_masked")
            if masked_as_gt:
                # write masked images as input images
                if os.path.exists(split_img_masked):
                    split_img_dir = split_img_masked
            else:
                # write masked images to seperate folder
                masked_images = False
                if os.path.exists(split_img_masked):
                    masked_images = True

            split_lbl_dir = os.path.join(root_dir, split, "labels")
            
            if not os.path.exists(split_img_dir) or not os.path.exists(split_lbl_dir):
                print(f"Warning: {split} directories not found in {root_dir}, skipping.")
                continue
            
            label_files = glob.glob(os.path.join(split_lbl_dir, "*.txt"))
            
            for label_path in tqdm(label_files, desc=f"Processing {split} from {os.path.basename(root_dir)}"):
                base_fname = os.path.basename(label_path).replace(".txt", "")
                
                # Try different image extensions (.png, .tif, .jpg, .jpeg)
                img_path = None
                for ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
                    potential_path = os.path.join(split_img_dir, base_fname + ext)
                    if os.path.exists(potential_path):
                        img_path = potential_path
                        fname_original = base_fname + ext
                        break
                
                if img_path is None:
                    print(f"Warning: image not found for label {label_path}")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: image not found or unreadable: {img_path}")
                    continue
                
                height, width = img.shape[:2]
                
                # Create unique filename to avoid conflicts between directories
                dir_name = os.path.basename(root_dir)
                original_ext = os.path.splitext(img_path)[1]
                unique_fname = f"{dir_name}_{base_fname}{original_ext}"
                
                # Copy image to output folder with unique name
                shutil.copy(img_path, os.path.join(output_img_dir, unique_fname))

                if masked_images and not masked_as_gt:
                    # also copy masked image
                    shutil.copy(os.path.join(split_img_masked, fname_original), os.path.join(output_masked_img_dir, unique_fname))
                
                # Add image entry
                images.append({
                    "id": image_id,
                    "file_name": unique_fname,
                    "width": width,
                    "height": height
                })
                
                # Read label file
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Warning: malformed line: {line}, skipping.")
                            continue
                        
                        class_id, x_c, y_c, w, h = map(float, parts)
                        class_id = int(class_id)
                        
                        # Check if class_id exists in the provided class dictionary
                        if class_id not in class_dict:
                            print(f"Warning: class_id {class_id} not found in class_dict, skipping annotation.")
                            continue
                        
                        # Convert normalized YOLO format to absolute COCO format
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
            "categories": categories
        }
        
        ann_file = os.path.join(output_dir, "annotations", f"instances_{split}.json")
        with open(ann_file, 'w') as f:
            json.dump(coco_dict, f, indent=2)
        
        print(f"Saved {split} annotations to {ann_file}")
        print(f"  - {len(images)} images")
        print(f"  - {len(annotations)} annotations")
    
    return

def data_to_coco_format_single(root_dir, output_dir):
    """
    Convert data from multiple root directories to COCO format.
    
    Args:
        root_directories: List of root directory paths, each containing train/val folders
        class_dict: Dictionary mapping class IDs to class names {id: name}
        output_dir: Output directory for the combined COCO dataset
    """
    # Create output directories
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

# list of classes in format for config files
def get_classes():

    class_file_1 = "/Stor1/wout/TreeAI4Species/ObjDet/5_RGB_S_320_pL/class5_RGB_all_L.xlsx"
    class_file_2 = "/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL/class12_RGB_all_L.xlsx"
    class_file_3 = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_a/34_RGB_ObjDet_640_pL/class34_RGB_PL.xlsx"
    class_file_4 = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_b/class34_RGB_PL_new.xlsx"

    classes_df_1 = pd.read_excel(class_file_1)
    classes_df_2 = pd.read_excel(class_file_2)
    classes_df_3 = pd.read_excel(class_file_3)
    classes_df_4 = pd.read_excel(class_file_4)

    # merge all and keep track of total train count for loss weights
    classes_df_12 = classes_df_1.merge(classes_df_2, how="outer", on=["Sp_ID", "Sp_Class"])
    classes_df_12["train"] = classes_df_12["train_x"].fillna(0) + classes_df_12["train_y"].fillna(0)
    classes_df_12 = classes_df_12.drop(["val_x", "val_y", "train_x", "train_y"], axis=1)
    classes_df_123 = classes_df_12.merge(classes_df_3, how="outer", on=["Sp_ID", "Sp_Class"])
    classes_df_123["train"] = classes_df_123["train_x"].fillna(0) + classes_df_123["train_y"].fillna(0)
    classes_df_123 = classes_df_123.drop(["val", "Label", "train_x", "train_y"], axis=1)
    classes_df_1234 = classes_df_123.merge(classes_df_4, how="outer", on=["Sp_ID", "Sp_Class"])
    classes_df_1234["train"] = classes_df_1234["train_x"].fillna(0) + classes_df_1234["train_y"].fillna(0)
    classes_df_1234 = classes_df_1234.drop(["val", "train_x", "train_y"], axis=1)

    # class list and map
    id_list = classes_df_1234["Sp_ID"].to_numpy()
    classes_list = classes_df_1234["Sp_Class"].to_numpy()
    classes_dict = {id_list[i]: classes_list[i] for i in range(len(id_list))}

    # class frequencies
    total_count = classes_df_1234["train"].sum()
    classes_df_1234["class_freq"] = classes_df_1234["train"] / total_count

    frequencies = classes_df_1234["class_freq"].to_numpy()

    return classes_list, classes_dict, frequencies


# get rgb and mean of dataset
def compute_rgb_mean_std(image_dir, exts=('.tif', '.png'), max_images=None):
    """
    Compute the per-channel RGB mean and standard deviation of all images in a directory.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing images.
    exts : tuple of str
        Valid image file extensions.
    max_images : int or None
        Maximum number of images to process (for speed). Use None to process all.

    Returns
    -------
    mean : tuple of float
        Mean of R, G, B channels.
    std : tuple of float
        Std of R, G, B channels.
    """

    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(image_dir, f'*{ext}')))

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        print("no images found")
        return

    # Accumulators
    channel_sum = np.zeros(3)
    channel_squared_sum = np.zeros(3)
    pixel_count = 0

    for path in tqdm(image_paths, desc="processing images"):
        with Image.open(path) as img:
            img = img.convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0  # normalize to [0, 1]
        
        h, w, c = img_np.shape
        pixel_count += h * w

        channel_sum += img_np.sum(axis=(0, 1))
        channel_squared_sum += (img_np ** 2).sum(axis=(0, 1))

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_squared_sum / pixel_count - mean ** 2)

    # back to range (0,255)
    return tuple(mean*255), tuple(std*255)


if __name__ == "__main__": 

    # get all classes, with mapping and frequencies

    # root_dir_1 = "/Stor1/wout/TreeAI4Species/ObjDet/12_RGB_ObjDet_640_fL"
    # root_dir_2 = "/Stor1/wout/TreeAI4Species/ObjDet/5_RGB_S_320_pL"
    # root_dir_3 = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_a/34_RGB_ObjDet_640_pL"
    # root_dir_4 = "/Stor1/wout/TreeAI4Species/ObjDet/34_RGB_ObjDet_640_pL_b"

    # root_dirs = [root_dir_1, root_dir_2, root_dir_3, root_dir_4]

    # classes_list, class_dict, class_frequencies = get_classes()

    # print(classes_list)
    # print(class_dict)
    # print(repr(class_frequencies))
    # class_weights = 1 / np.log(1.02 + class_frequencies)
    # print(repr(class_weights))

    # convert to coco format

    # odir = "/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0"

    # data_to_coco_format(root_dirs=root_dirs, class_dict=class_dict, output_dir=odir)

    # get mean and std of training set
    # img_dir = "/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0/train/images/"
    img_dir = "/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/12_RGB_ObjDet_640_fL/train/images/"
    mean, std = compute_rgb_mean_std(img_dir)

    print(f"\nRGB Mean: {mean}")
    print(f"RGB Std:  {std}")
