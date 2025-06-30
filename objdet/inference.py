import os
import glob
import json
import cv2
from tqdm import tqdm
import argparse

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmengine.config import Config, ConfigDict


class_dict = {3: 'picea abies', 6: 'pinus sylvestris', 9: 'larix decidua', 12: 'fagus sylvatica', 13: 'dead tree', 24: 'abies alba', 26: 'pseudotsuga menziesii', 30: 'acer pseudoplatanus', 36: 'fraxinus excelsior', 43: 'acer sp.', 49: 'tilia cordata', 56: 'quercus sp.', 62: 'Tilia platyphyllos', 63: 'Tilia spp', 64: 'Ulmus glabra', 1: 'betula papyrifera', 2: 'tsuga canadensis', 4: 'acer saccharum', 5: 'betula sp.', 7: 'picea rubens', 8: 'betula alleghaniensis', 10: 'fagus grandifolia', 11: 'picea sp.', 14: 'acer pensylvanicum', 15: 'populus balsamifera', 16: 'quercus ilex', 17: 'quercus robur', 18: 'pinus strobus', 19: 'larix laricina', 20: 'larix gmelinii', 21: 'pinus pinea', 22: 'populus grandidentata', 23: 'pinus montezumae', 25: 'betula pendula', 27: 'fraxinus nigra', 28: 'dacrydium cupressinum', 29: 'cedrus libani', 31: 'pinus elliottii', 32: 'cryptomeria japonica', 33: 'pinus koraiensis', 34: 'abies holophylla', 35: 'alnus glutinosa', 37: 'coniferous', 38: 'eucalyptus globulus', 39: 'pinus nigra', 40: 'quercus rubra', 41: 'tilia europaea', 42: 'abies firma', 44: 'metrosideros umbellata', 45: 'acer rubrum', 46: 'picea mariana', 47: 'abies balsamea', 48: 'castanea sativa', 50: 'populus sp.', 51: 'crataegus monogyna', 52: 'quercus petraea', 53: 'acer platanoides', 61: 'salix sp.', 60: 'deciduous', 54: 'robinia pseudoacacia', 58: 'pinus sp.', 57: 'salix alba', 59: 'carpinus sp.'}


def setup_tta_config_fixed(config_path):
    cfg = Config.fromfile(config_path)
    
    cfg.tta_model = dict(
        type='DetTTAModel',
        tta_cfg=dict(
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=100
        )
    )
    
    # img_scales = [
    #     (640, 640),
    #     (576, 576),
    #     (704, 704),
    # ]
    # cfg.tta_pipeline = [
    #     dict(type='LoadImageFromFile',
    #      backend_args=None),
    #     dict(
    #         type='TestTimeAug',
    #         transforms=[
    #             [
    #                 dict(type='Resize', scale=scale, keep_ratio=True)
    #                 for scale in img_scales
    #             ],
    #             [
    #                 dict(type='RandomFlip', prob=0.0),  # No flip
    #                 dict(type='RandomFlip', prob=1.0, direction='horizontal'),  # Horizontal flip
    #             ],
    #             [dict(type='PackDetInputs')]
    #         ]
    #     )
    # ]
    # try only flips
    # resize seems to do weird things
    img_scale = (640, 640)
    cfg.tta_pipeline = [
        dict(type='LoadImageFromFile',
         backend_args=None),
        dict(
            type='TestTimeAug',
            transforms=[
                [dict(type='Resize', scale=img_scale, keep_ratio=True)],
                [
                    dict(type='RandomFlip', prob=0.0),  # No flip
                    dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                    dict(type='RandomFlip', prob=1.0, direction='vertical'),
                ],
                [dict(type='PackDetInputs')]
            ]
        )
    ]
    
    # mmdetection weirdness
    cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
    test_data_cfg = cfg.test_dataloader.dataset
    while 'dataset' in test_data_cfg:
        test_data_cfg = test_data_cfg['dataset']
    test_data_cfg.pipeline = cfg.tta_pipeline
    return cfg

def setup_tta_config(config_path):
    """Setup TTA configuration"""
    cfg = Config.fromfile(config_path)
    
    # Setup TTA model wrapper
    cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
    
    img_scales = [(512, 512), (768, 768)]

    cfg.tta_pipeline = [
    dict(type='LoadImageFromFile',
         backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=s, keep_ratio=True) for s in img_scales
        ], [
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs')
       ]])
    ]

    # mmdetection weirdness, copied from demo script
    cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
    test_data_cfg = cfg.test_dataloader.dataset
    while 'dataset' in test_data_cfg:
        test_data_cfg = test_data_cfg['dataset']

    test_data_cfg.pipeline = cfg.tta_pipeline

    return cfg

def run_inference(config_path, checkpoint_path, test_images_dir, output_dir, 
                  use_tta=True, conf_threshold=0.3, save_visualizations=True):
    """
    Run inference on test images with optional TTA
    
    Args:
        config_path: Path to model config file
        checkpoint_path: Path to trained model checkpoint
        test_images_dir: Directory containing test images
        output_dir: Directory to save results
        use_tta: Whether to use test time augmentation
        conf_threshold: Confidence threshold for detections
        save_visualizations: Whether to save visualization images
    """
    
    # Register all modules
    register_all_modules()
    
    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    if save_visualizations:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Load config and setup TTA if needed
    if use_tta:
        print("Setting up Test Time Augmentation")
        cfg = setup_tta_config_fixed(config_path)
        model = init_detector(cfg, checkpoint_path, device='cuda:0', cfg_options={})
    else:
        cfg = Config.fromfile(config_path)
        model = init_detector(cfg, checkpoint_path, device='cuda:0')
    
    # Initialize model
    print(f"Loading model from {checkpoint_path}...")
    
    # Find all test images
    image_extensions = ['*.png', '*.tif', '*.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))
    
    print(f"Found {len(test_images)} test images")

    classes = list(class_dict.values())
    
    # Results storage
    all_results = []
    
    # Run inference
    for img_path in tqdm(test_images, desc="Running inference"):
        img_name = os.path.basename(img_path)
        
        # Run detection
        result = inference_detector(model, img_path)
        
        # Extract predictions
        pred_instances = result.pred_instances
        bboxes = pred_instances.bboxes.cpu().numpy()
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()
        
        # Filter by confidence threshold
        keep_idx = scores >= conf_threshold
        bboxes = bboxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]
        
        # Store results
        img_results = {
            'image_name': img_name,
            'detections': []
        }
        
        for bbox, score, label in zip(bboxes, scores, labels):
            detection = {
                'bbox': bbox.tolist(),  # [x1, y1, x2, y2]
                'score': float(score),
                # 'class_id': int(label),
                'class_id': int(label)+1, # new version: labels already converted back but of by one
                'class_name': classes[label]
            }
            img_results['detections'].append(detection)
        
        all_results.append(img_results)
        
        # Save visualization if requested
        if save_visualizations and len(bboxes) > 0:
            img = cv2.imread(img_path)
            for bbox, score, label in zip(bboxes, scores, labels):
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label and score
                label_text = f"{label} ({classes[label]}): {score:.2f}"
                cv2.putText(img, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save visualization
            vis_path = os.path.join(output_dir, 'visualizations', f"vis_{img_name}")
            cv2.imwrite(vis_path, img)
    
    # save results to json
    results_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # save summary  
    total_detections = sum(len(r['detections']) for r in all_results)
    avg_detections = total_detections / len(all_results) if all_results else 0
    summary = {
        'total_images': len(all_results),
        'total_detections': total_detections,
        'avg_detections_per_image': avg_detections,
        'confidence_threshold': conf_threshold,
        'used_tta': use_tta,
        'class_distribution': {}
    }
    
    # Calculate class distribution
    for result in all_results:
        for det in result['detections']:
            class_name = det['class_name']
            summary['class_distribution'][class_name] = summary['class_distribution'].get(class_name, 0) + 1
    
    summary_path = os.path.join(output_dir, 'inference_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nInference completed!")
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    
    if save_visualizations:
        print(f"Visualizations saved to: {os.path.join(output_dir, 'visualizations')}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference script')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--test_dir', required=True, help='Directory containing test images')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--use_tta', action='store_true', help='Use test time augmentation')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--no_vis', action='store_true', help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_images_dir=args.test_dir,
        output_dir=args.output_dir,
        use_tta=args.use_tta,
        conf_threshold=args.conf_threshold,
        save_visualizations=not args.no_vis
    )