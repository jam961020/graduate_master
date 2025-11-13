"""
Extract CLIP-based environment features for all images
Uses YOLO ROI + CLIP semantic encoding
"""

import json
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from clip_environment import CLIPEnvironmentEncoder
from yolo_detector import YOLODetector


def extract_clip_features_all_images(image_dir, gt_file, yolo_model_path, output_file):
    """
    Extract CLIP features for all images in dataset
    
    Args:
        image_dir: Path to images directory
        gt_file: Ground truth JSON file
        yolo_model_path: Path to YOLO model
        output_file: Output JSON file path
    """
    print("="*70)
    print("CLIP Environment Feature Extraction")
    print("="*70)
    
    # 1. Initialize CLIP encoder
    print("\n[1] Initializing CLIP encoder...")
    clip_encoder = CLIPEnvironmentEncoder()
    
    # 2. Initialize YOLO detector
    print("\n[2] Initializing YOLO detector...")
    yolo_detector = YOLODetector(yolo_model_path)
    
    # 3. Load ground truth
    print("\n[3] Loading ground truth...")
    with open(gt_file) as f:
        gt_data = json.load(f)
    
    image_names = list(gt_data.keys())
    print(f"  Found {len(image_names)} images in GT")
    
    # 4. Extract features
    print("\n[4] Extracting CLIP features from ROI...")
    
    image_dir = Path(image_dir)
    results = {}
    
    for img_name in tqdm(image_names, desc="Processing"):
        # Find image file
        img_path = image_dir / f"{img_name}.jpg"
        if not img_path.exists():
            img_path = image_dir / f"{img_name}.png"
        
        if not img_path.exists():
            print(f"  [WARN] Image not found: {img_name}")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Failed to load: {img_name}")
            continue
        
        # Detect ROI with YOLO
        try:
            rois = yolo_detector.detect_rois(image)  # [(class_id, x1, y1, x2, y2), ...]

            if len(rois) == 0:
                # No ROI detected, use full image
                print(f"  [WARN] No ROI detected for {img_name}, using full image")
                roi_crop = image
            else:
                # Prefer longi_WL (class 2) for welding line ROI
                # If not found, use first detection
                longi_roi = [roi for roi in rois if roi[0] == 2]

                if longi_roi:
                    _, x1, y1, x2, y2 = longi_roi[0]
                else:
                    _, x1, y1, x2, y2 = rois[0]

                roi_crop = image[y1:y2, x1:x2]

                if roi_crop.size == 0:
                    print(f"  [WARN] Empty ROI for {img_name}, using full image")
                    roi_crop = image

        except Exception as e:
            print(f"  [ERROR] YOLO failed for {img_name}: {e}")
            import traceback
            traceback.print_exc()
            roi_crop = image
        
        # Extract CLIP features
        try:
            clip_features = clip_encoder.encode_roi(roi_crop)
            
            # Convert to dict
            feature_dict = {}
            for name, value in zip(clip_encoder.get_feature_names(), clip_features):
                feature_dict[name] = float(value)
            
            results[img_name] = feature_dict
        
        except Exception as e:
            print(f"  [ERROR] CLIP encoding failed for {img_name}: {e}")
            continue
    
    # 5. Save results
    print(f"\n[5] Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Done! Extracted features for {len(results)}/{len(image_names)} images")
    
    # 6. Statistics
    print(f"\n[6] Feature Statistics:")
    feature_names = clip_encoder.get_feature_names()
    
    for fname in feature_names:
        values = [results[img][fname] for img in results if fname in results[img]]
        if values:
            print(f"  {fname:<20}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../dataset/images/test",
                       help="Image directory")
    parser.add_argument("--gt_file", default="../dataset/ground_truth.json",
                       help="Ground truth JSON")
    parser.add_argument("--yolo_model", default="models/best.pt",
                       help="YOLO model path")
    parser.add_argument("--output", default="environment_clip.json",
                       help="Output JSON file")
    args = parser.parse_args()
    
    results = extract_clip_features_all_images(
        args.image_dir,
        args.gt_file,
        args.yolo_model,
        args.output
    )


if __name__ == "__main__":
    main()
