import time
import json 
import copy

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def convert_predictions_txt_to_coco(predictions_txt_path, image_name_to_size, image_name_to_id):
    """
    Convert predictions.txt to COCO detection format.

    Parameters
    ----------
    predictions_txt_path : str
        Path to the predictions.txt file.
        Each line should be: image_id class_id x_center y_center width height score (all space-separated).
        Coordinates are in fractional image size (0–1).
    image_id_to_size : dict
        Dictionary mapping image_id (int) to (width, height) in pixels.

    Returns
    -------
    list of dict
        COCO-style detection list for evaluation.
    """
    coco_results = []

    # SCORE_THRESH = 0.25
    # no filtering
    SCORE_THRESH = 0

    with open(predictions_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 7:
                print(f"Skipping malformed line: {line}")
                continue

            image_name = parts[0]
            category_id = int(parts[1])
            x_center, y_center, width, height, score = map(float, parts[2:])

            if score < SCORE_THRESH:
                continue

            if image_name not in image_name_to_size:
                raise ValueError(f"Image ID {image_name} not found in image_id_to_size mapping.")

            img_w, img_h = image_name_to_size[image_name]
            image_id = image_name_to_id[image_name]


            # Convert fractional coords to pixel absolute coords
            abs_cx = x_center * img_w
            abs_cy = y_center * img_h
            abs_w = width * img_w
            abs_h = height * img_h

            x1 = abs_cx - abs_w / 2
            y1 = abs_cy - abs_h / 2

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, abs_w, abs_h],
                "score": score
            })

    return coco_results

def image_name_to_id_size(coco_gt):
    name_to_size = {img["file_name"][:-4]: (img["width"], img["height"]) for img in coco_gt.dataset["images"]}
    name_to_id = {img["file_name"][:-4]: img["id"] for img in coco_gt.dataset["images"]}
    return name_to_size, name_to_id

def evaluate_map(gt_json_path, pred_json_path, iou_type="bbox"):
    """
    Evaluate mAP between COCO ground truth and prediction files.

    Parameters
    ----------
    gt_json_path : str
        Path to ground truth annotations in COCO format.
    pred_json_path : str
        Path to prediction results in COCO detection format (list of dicts).
    iou_type : str
        Type of evaluation: 'bbox', 'segm', or 'keypoints'.
    """
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":

    gt_path = "/Stor1/wout/TreeAI4Species/ObjDet/converted_coco/all_no0_masked_images_as_gt/annotations/instances_val.json"

    dir_output = "/Stor1/wout/TreeAI4Species/val_predictions/new_subm_test3/"
    # pred_path = dir_output + "predictions.txt"
    pred_path = dir_output + "predictions_postprocess.txt"

    out_path_json = dir_output + 'predictions_coco.json'

    coco_gt = COCO(gt_path)
    name_to_size, name_to_id = image_name_to_id_size(coco_gt)

    pred_json = convert_predictions_txt_to_coco(pred_path, name_to_size, name_to_id)

    with open(out_path_json, "w") as f:
        json.dump(pred_json, f)

    evaluate_map(gt_path, out_path_json)
