import os
import glob
import cv2
import json

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from util import xy_to_img_coordinates, label_file_to_list

from ensemble_boxes import weighted_boxes_fusion

class_dict = {3: 'picea abies', 6: 'pinus sylvestris', 9: 'larix decidua', 12: 'fagus sylvatica', 13: 'dead tree', 24: 'abies alba', 26: 'pseudotsuga menziesii', 30: 'acer pseudoplatanus', 36: 'fraxinus excelsior', 43: 'acer sp.', 49: 'tilia cordata', 56: 'quercus sp.', 62: 'Tilia platyphyllos', 63: 'Tilia spp', 64: 'Ulmus glabra', 1: 'betula papyrifera', 2: 'tsuga canadensis', 4: 'acer saccharum', 5: 'betula sp.', 7: 'picea rubens', 8: 'betula alleghaniensis', 10: 'fagus grandifolia', 11: 'picea sp.', 14: 'acer pensylvanicum', 15: 'populus balsamifera', 16: 'quercus ilex', 17: 'quercus robur', 18: 'pinus strobus', 19: 'larix laricina', 20: 'larix gmelinii', 21: 'pinus pinea', 22: 'populus grandidentata', 23: 'pinus montezumae', 25: 'betula pendula', 27: 'fraxinus nigra', 28: 'dacrydium cupressinum', 29: 'cedrus libani', 31: 'pinus elliottii', 32: 'cryptomeria japonica', 33: 'pinus koraiensis', 34: 'abies holophylla', 35: 'alnus glutinosa', 37: 'coniferous', 38: 'eucalyptus globulus', 39: 'pinus nigra', 40: 'quercus rubra', 41: 'tilia europaea', 42: 'abies firma', 44: 'metrosideros umbellata', 45: 'acer rubrum', 46: 'picea mariana', 47: 'abies balsamea', 48: 'castanea sativa', 50: 'populus sp.', 51: 'crataegus monogyna', 52: 'quercus petraea', 53: 'acer platanoides', 61: 'salix sp.', 60: 'deciduous', 54: 'robinia pseudoacacia', 58: 'pinus sp.', 57: 'salix alba', 59: 'carpinus sp.'}

# post process to remove bboxs with very high overlap but different class
def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_width = max(0.0, inter_x_max - inter_x_min)
    inter_height = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0.0:
        return 0.0  # Avoid division by zero

    return inter_area / union_area

def post_process_results(predictions_file, out_path):
    # if overlapping bboxs (IoU higher then 0.85) from different class (not handled by nms or wbf, occurs sometimes), take one with the highest confidence

    IOU_THRESH = 0.85
    
    # get df of all bboxs
    all_bboxs = label_file_to_list(predictions_file, submission_output=True)

    all_image_ids = np.unique(all_bboxs["img_id"].to_numpy())

    post_processed_lines = []
    for img_id in all_image_ids:

        bboxs_image = all_bboxs[all_bboxs["img_id"] == img_id].to_numpy()

        remove_indices = []
        for i, bbox_i in enumerate(bboxs_image):
            for j, bbox_j in enumerate(bboxs_image):

                if j <= i:
                    # only calculate forward in list, iou is symmetric
                    continue

                if j in remove_indices:
                    # j is already on remove list, skip
                    continue

                bbox_i = bboxs_image[i,2:6]
                bbox_j = bboxs_image[j,2:6]

                iou = calculate_iou(bbox_i, bbox_j)

                if iou > IOU_THRESH:
                    # keep bbox with highest confidence
                    conf_i = bboxs_image[i,-1]
                    conf_j = bboxs_image[j, -1]
                    if conf_j > conf_i:
                        remove_indices.append(i)
                    else:
                        remove_indices.append(j)
                    # assume only one bbox with this overlap
                    break

        bboxs_keep = np.delete(bboxs_image, remove_indices, axis=0)
        for bbox in bboxs_keep:
            line = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]} {bbox[6]}"
            post_processed_lines.append(line)

    with open(out_path, 'w') as f:
        for line in post_processed_lines:
            f.write(f"{line}\n")
    print(f"Saved postprocessed submission .txt to {out_path}")


# read in converted predictions from multiple inference runs and combine
def ensemble_predictions(predictions_file1, predictions_file2, output_path, weights=None, ):
    # get df of all bboxs
    all_bboxs1 = label_file_to_list(predictions_file1, submission_output=True)
    all_bboxs2 = label_file_to_list(predictions_file2, submission_output=True)

    all_image_ids = np.unique(all_bboxs1["img_id"].to_numpy())
    
    ensembled_predictions_lines = []
    for img_id in all_image_ids:

        bboxs_image1 = all_bboxs1[all_bboxs1["img_id"] == img_id]
        bboxs_image2 = all_bboxs2[all_bboxs2["img_id"] == img_id]


        boxes_list = []
        scores_list = []
        labels_list = []

        # construct bboxs with wbf ready format
        for index, row in bboxs_image1.iterrows():
            x1 = row["x_center"] - row["width"]/2
            y1 = row["y_center"] - row["height"]/2
            x2 = row["x_center"] + row["width"]/2
            y2 = row["y_center"] + row["height"]/2
            score = row["conf"]
            label = row["species"]

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(score)
            labels_list.append(label)
        
        for index, row in bboxs_image2.iterrows():
            x1 = row["x_center"] - row["width"]/2
            y1 = row["y_center"] - row["height"]/2
            x2 = row["x_center"] + row["width"]/2
            y2 = row["y_center"] + row["height"]/2
            score = row["conf"]
            label = row["species"]

            boxes_list.append([x1, y1, x2, y2])
            scores_list.append(score)
            labels_list.append(label)
    
        # apply wbf
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, 
            scores_list, 
            labels_list,
            weights=weights,  # should be list, based on MAP of individual models
            iou_thr=0.6,
            skip_box_thr=0,
            conf_type='avg'
        )

        for i, box in enumerate(boxes):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = abs(box[2]-box[0])
            height = abs(box[3] - box[1])
            ensembled_line = f"{img_id} {labels[i]} {x_center} {y_center} {width} {height} {scores[i]}"
            ensembled_predictions_lines.append(ensembled_line)

    with open(output_path, 'w') as f:
        for line in ensembled_predictions_lines:
            f.write(f"{line}\n")
    print(f"Saved final submission .txt to {output_path}")
    
    return



# results of coco model are actually written with index as label, must be converted back to species ID via classes mapping
def convert_results(results_path, img_dir, output_path):
    """
    Convert results (json from inference.py) to competition format
    """

    with open(results_path, "r") as f:
        results_dict = json.load(f)

    species_ids = list(class_dict)

    all_results_lines = []
    for result in tqdm(results_dict, desc="converting results"):

        img_id = os.path.splitext(result["image_name"])[0]

        img_path = os.path.join(img_dir, result["image_name"])
        img = cv2.imread(img_path)
        img_width, img_height, _ = img.shape

        for detection in result["detections"]:
            class_id = detection["class_id"]
            # species ID is actually different from class ID!! class ID is label determined by COCO, which is converted to start at 0 in the order that categories is given in the annotation.json file
            # so to convert back, just index the same dict that is in the categories file and use the key
            # species = species_ids[class_id]
            # NEW VERSION: ALREADY CORRECT!
            species = class_id
            conf = detection["score"]
            # get center coords and width and height of preds
            bbox = detection["bbox"]
            x_min, y_min = bbox[0:2]
            x_max, y_max = bbox[2:4]
            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width/2
            y_center = y_min + height/2
            # convert to fractional coords
            x_frac = x_center/img_width
            y_frac = y_center/img_height
            width_frac = width/img_width
            height_frac = height/img_height

            line = f"{img_id} {species} {x_frac} {y_frac} {width_frac} {height_frac} {conf}"
            all_results_lines.append(line)

    
    # write
    with open(output_path, 'w') as f:
        for line in all_results_lines:
            f.write(f"{line}\n")
    print(f"Saved final submission .txt to {output_path}")

    return

def vis_submission(img_dir, submission_path, out_path):

    image_paths = glob.glob(os.path.join(img_dir, "*.png"))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # get df of all bboxs
    all_bboxs = label_file_to_list(submission_path, submission_output=True)

    for image_path in image_paths:
        img_id = os.path.splitext(os.path.basename(image_path))[0]
        img = cv2.imread(image_path)

        bboxs_img = all_bboxs[all_bboxs["img_id"] == img_id]

        # convert back to image coords
        bboxs_converted = xy_to_img_coordinates(bboxs_img, img)

        # plot
        fig, ax = plt.subplots()
        ax.imshow(img)
        for _, row in bboxs_converted.iterrows():
            x = row["x_center"] - row["width"]/2
            y = row["y_center"] - row["height"]/2
            w = row["width"]
            h = row["height"]
            # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            rect = patches.Rectangle((x,y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x,y, f"{int(row['species'])} ({row['conf']:.2f})", fontsize=8, color="white", bbox=dict(facecolor='red', edgecolor="none"))
            
        fig.savefig(os.path.join(out_path, f"vis_subm_{img_id}.png"))
        plt.close()

    return


def filter_on_score(predictions_path, out_path, confidence_th):

    predictions = label_file_to_list(predictions_path, submission_output=True)

    predictions_filtered = predictions[predictions["conf"] >= confidence_th]
    
    lines_out = [f"{row[0]} {row[1]} {row[2]} {row[3]} {row[4]} {row[5]} {row[6]}" for _,row in predictions_filtered.iterrows()]
    with open(out_path, 'w') as f:
        for line in lines_out:
            f.write(f"{line}\n")
    print(f"Saved filtered submission .txt to {out_path}")

    return



if __name__ == "__main__":

    img_dir = "/Stor1/wout/TreeAI4Species/test_submission/12_RGB_ObjDet_640_fL_test_images/images/"

    output_path = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_resumed_aug_normalize/"
    results_path = os.path.join(output_path, "inference_results.json")
    out_path_predictions = os.path.join(output_path, "predictions.txt")
    out_path_predictions_vis = os.path.join(output_path, "submission_vis")

    convert_results(results_path, img_dir, out_path_predictions)

    # vis_submission(img_dir, out_path_predictions, out_path_predictions_vis)

    out_path_postprocess = os.path.join(output_path, "predictions_postprocess.txt")
    post_process_results(out_path_predictions, out_path_postprocess)


    out_path_final = os.path.join(output_path, "predictions_final.txt")
    filter_on_score(out_path_postprocess, out_path_final, confidence_th=0.1)

    vis_submission(img_dir, out_path_final, out_path_predictions_vis)

    # for val eval
    # img_dir = "/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0_masked_images_as_gt/val/images/"

    # output_path = "/Stor1/wout/TreeAI4Species/val_predictions/new_subm_test3/"
    # results_path = os.path.join(output_path, "inference_results.json")
    # out_path_predictions = os.path.join(output_path, "predictions.txt")
    # out_path_predictions_vis = os.path.join(output_path, "submission_vis")

    # # convert_results(results_path, img_dir, out_path_predictions)

    # # vis_submission(img_dir, out_path_predictions, out_path_predictions_vis)

    # out_path_postprocess = os.path.join(output_path, "predictions_postprocess.txt")
    # post_process_results(out_path_predictions, out_path_postprocess)


    # post_process_folder = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_no_filtering_1/"
    # in_path_predictions = post_process_folder + "predictions.txt"
    # out_path_post_processed = post_process_folder + "predictions_postprocessed.txt"
    # out_path_vis_postprocessed = post_process_folder + "submission_vis_post_process"

    # post_process_results(in_path_predictions, out_path_post_processed)

    # vis_submission(img_dir, out_path_post_processed, out_path_vis_postprocessed)


    # out_path1 = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_no_filtering_1/"
    # predictions1 = os.path.join(out_path1, "predictions.txt")

    # out_path2 = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_resumed_weighted_sample_ep63/"
    # predictions2 = os.path.join(out_path2, "predictions.txt")

    # out_dir = "/Stor1/wout/TreeAI4Species/output_submission/ensemble_orig_resumed_ep64"
    # out_path_predictions = os.path.join(out_dir, "predictions.txt")

    # weights = (0.75, 0.25)

    # ensemble_predictions(predictions1, predictions2, out_path_predictions)

    # out_path_predictions_vis = os.path.join(out_dir, "submission_vis")
