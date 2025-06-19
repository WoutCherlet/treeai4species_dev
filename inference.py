import os
import glob
import json
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmcv import Config
import torch


def setup_tta_config(config_path, tta_transforms=None):
    """Setup TTA configuration"""
    cfg = Config.fromfile(config_path)
    
    if tta_transforms is None:
        # Default TTA transforms
        tta_transforms = [
            # Original image
            dict(type='LoadImageFromFile'),
            # Horizontal flip
            [
                dict(type='LoadImageFromFile'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal')
            ],
            # Multi-scale
            [
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(512, 512), keep_ratio=True)
            ],
            [
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(768, 768), keep_ratio=True)
            ],
            # Combined transforms
            [
                dict(type='LoadImageFromFile'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                dict(type='Resize', scale=(512, 512), keep_ratio=True)
            ],
            [
                dict(type='LoadImageFromFile'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
                dict(type='Resize', scale=(768, 768), keep_ratio=True)
            ]
        ]
    
    # Setup TTA model wrapper
    cfg.model = dict(
        type='DetTTAWrapper',
        detector=cfg.model,
        test_cfg=dict(
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
    
    # TTA pipeline
    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='TestTimeAug',
            transforms=tta_transforms
        )
    ]
    
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
        cfg = setup_tta_config(config_path)
    else:
        cfg = Config.fromfile(config_path)
    
    # Initialize model
    print(f"Loading model from {checkpoint_path}...")
    model = init_detector(cfg, checkpoint_path, device='cuda:0')
    
    # Get class names from config
    classes = cfg.metainfo['classes']
    
    # Find all test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext.upper())))
    
    print(f"Found {len(test_images)} test images")
    
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
                'class_id': int(label),
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
                label_text = f"{classes[label]}: {score:.2f}"
                cv2.putText(img, label_text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save visualization
            vis_path = os.path.join(output_dir, 'visualizations', f"vis_{img_name}")
            cv2.imwrite(vis_path, img)
    
    # Save results to JSON
    results_path = os.path.join(output_dir, 'inference_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary statistics  
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

def post_process_results():
    # TODO: post processing of results here

    # in public test dataset there are full black and white areas which are masked out, any bboxs that include these areas can be filtered out or cropped!

    raise NotImplementedError

def convert_results(results, output_path):
    """
    Convert results to COCO format for evaluation
    
    Args:
        results: Results from run_inference
        output_path: Path to save predictions.txt file at
        image_info_dict: Optional dict with image dimensions {img_name: (width, height)}
    """
    all_results_lines = []
    
    # TODO: check output format results
    for img_idx, result in enumerate(results):
        # TODO: read bbox here and convert to format
        # pseudocode:
        '''
        for bbox in result:
            convert to fractional coords
            width = w/img_width
            height = h/img_height
            x_center = x/img_width + width/2
            y_center = y/img_height + height/2
            conf = bbox.conf
            species = bbox.cls
        
            line = f"{img_id} {species} {x_center} {y_center} {width} {height} {conf}
        '''

        continue

    # write
    with open(os.path.join(output_path, "predictions.txt"), 'w') as f:
        for line in all_results_lines:
            f.write(f"{line}\n")
        
    
    
    print(f"Saved final submission .txt to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with MMDetection model')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--test-dir', required=True, help='Directory containing test images')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--no-tta', action='store_true', help='Disable test time augmentation')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--no-vis', action='store_true', help='Skip saving visualizations')
    
    args = parser.parse_args()
    
    # Run inference
    results = run_inference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        test_images_dir=args.test_dir,
        output_dir=args.output_dir,
        use_tta=not args.no_tta,
        conf_threshold=args.conf_threshold,
        save_visualizations=not args.no_vis
    )

    converted_output_path = os.path.join(args.output_dir, "")
    
    # Optionally convert to COCO format
    coco_output_path = os.path.join(args.output_dir, 'coco_format_results.json')
    convert_results(results, coco_output_path)