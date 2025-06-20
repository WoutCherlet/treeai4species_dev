import os
import glob
import cv2
import json

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import patches

from util import xy_to_img_coordinates, label_file_to_list

class_dict = {3: 'picea abies', 6: 'pinus sylvestris', 9: 'larix decidua', 12: 'fagus sylvatica', 13: 'dead tree', 24: 'abies alba', 26: 'pseudotsuga menziesii', 30: 'acer pseudoplatanus', 36: 'fraxinus excelsior', 43: 'acer sp.', 49: 'tilia cordata', 56: 'quercus sp.', 62: 'Tilia platyphyllos', 63: 'Tilia spp', 64: 'Ulmus glabra', 1: 'betula papyrifera', 2: 'tsuga canadensis', 4: 'acer saccharum', 5: 'betula sp.', 7: 'picea rubens', 8: 'betula alleghaniensis', 10: 'fagus grandifolia', 11: 'picea sp.', 14: 'acer pensylvanicum', 15: 'populus balsamifera', 16: 'quercus ilex', 17: 'quercus robur', 18: 'pinus strobus', 19: 'larix laricina', 20: 'larix gmelinii', 21: 'pinus pinea', 22: 'populus grandidentata', 23: 'pinus montezumae', 25: 'betula pendula', 27: 'fraxinus nigra', 28: 'dacrydium cupressinum', 29: 'cedrus libani', 31: 'pinus elliottii', 32: 'cryptomeria japonica', 33: 'pinus koraiensis', 34: 'abies holophylla', 35: 'alnus glutinosa', 37: 'coniferous', 38: 'eucalyptus globulus', 39: 'pinus nigra', 40: 'quercus rubra', 41: 'tilia europaea', 42: 'abies firma', 44: 'metrosideros umbellata', 45: 'acer rubrum', 46: 'picea mariana', 47: 'abies balsamea', 48: 'castanea sativa', 50: 'populus sp.', 51: 'crataegus monogyna', 52: 'quercus petraea', 53: 'acer platanoides', 61: 'salix sp.', 60: 'deciduous', 54: 'robinia pseudoacacia', 58: 'pinus sp.', 57: 'salix alba', 59: 'carpinus sp.'}


def post_process_results():
    # TODO: post processing of results here

    # in public test dataset there are full black and white areas which are masked out, any bboxs that include these areas can be filtered out or cropped!
    # seems like model does this quite well by itself

    raise NotImplementedError


# TODO: results of coco model are actually written with index as label, must be converted back to species ID via classes mapping
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
            species = species_ids[class_id]
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


if __name__ == "__main__":

    results_path = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_no_filtering_1/inference_results.json"
    img_dir = "/Stor1/wout/TreeAI4Species/test_submission/12_RGB_ObjDet_640_fL_test_images/images/"
    out_path_predictions = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_no_filtering_1/predictions.txt"
    out_path_predictions_vis = "/Stor1/wout/TreeAI4Species/output_submission/allno0_masked_as_gt_no_filtering_1/submission_vis"

    convert_results(results_path, img_dir, out_path_predictions)

    vis_submission(img_dir, out_path_predictions, out_path_predictions_vis)

