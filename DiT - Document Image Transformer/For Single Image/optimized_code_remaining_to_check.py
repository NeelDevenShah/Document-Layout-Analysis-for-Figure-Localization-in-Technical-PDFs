import os
import sys

# Only install dependencies if not already present
if not os.path.exists('detectron2'):
    os.system('git clone --depth 1 https://github.com/facebookresearch/detectron2.git')
    os.system('pip install -e detectron2')

if not os.path.exists('unilm'):
    os.system("git clone --depth 1 https://github.com/microsoft/unilm.git")
    os.system("sed -i 's/from collections import Iterable/from collections.abc import Iterable/' unilm/dit/object_detection/ditod/table_evaluation/data_structure.py")

sys.path.append("unilm")
sys.path.append("detectron2")

import cv2
import torch
from huggingface_hub import hf_hub_download
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from unilm.dit.object_detection.ditod import add_vit_config

def setup_config(confidence_threshold=0.7):
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file("cascade_dit_base.yml")
    
    if not os.path.exists('publaynet_dit-b_cascade.pth'):
        filepath = hf_hub_download(
            repo_id="Sebas6k/DiT_weights",
            filename="publaynet_dit-b_cascade.pth",
            repo_type="model"
        )
        cfg.MODEL.WEIGHTS = filepath
    else:
        cfg.MODEL.WEIGHTS = 'publaynet_dit-b_cascade.pth'
    
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def analyze_image(original_img, predictor, confidence_threshold=0.7):
    md = MetadataCatalog.get('publaynet_val')
    md.set(thing_classes=["text", "title", "list", "table", "figure"])

    # Store original dimensions
    original_height, original_width = original_img.shape[:2]
    
    # Create a copy of the image for processing
    img = original_img.copy()
    
    # Calculate resize scale if needed
    max_size = 1333
    scale = 1.0
    if max(original_height, original_width) > max_size:
        scale = max_size / max(original_height, original_width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Get predictions
    with torch.no_grad():
        outputs = predictor(img)["instances"]

    # Filter by confidence
    mask = outputs.scores > confidence_threshold
    outputs = outputs[mask]

    # Visualize results (on resized image)
    v = Visualizer(img[:, :, ::-1],
                   md,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(outputs.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # Extract high-confidence figure bounding boxes and scale back to original size
    figure_bboxes = []
    for i in range(len(outputs)):
        if outputs.pred_classes[i] == md.thing_classes.index("figure"):
            # Get the bbox coordinates
            bbox = outputs.pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            
            # Scale coordinates back to original image size if image was resized
            if scale != 1.0:
                bbox = [
                    bbox[0] / scale,  # x1
                    bbox[1] / scale,  # y1
                    bbox[2] / scale,  # x2
                    bbox[3] / scale   # y2
                ]
            
            confidence = float(outputs.scores[i])
            figure_bboxes.append({
                'bbox': bbox,
                'confidence': confidence
            })

    # Scale result image back to original size for visualization
    if scale != 1.0:
        result_image = cv2.resize(result_image, (original_width, original_height))

    return result_image, figure_bboxes

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process an image with optimized layout detection.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0-1)')
    args = parser.parse_args()

    # Initialize model with optimizations
    cfg = setup_config(args.confidence)
    predictor = DefaultPredictor(cfg)

    # Process image
    original_img = cv2.imread(args.image_path)
    if original_img is None:
        raise ValueError(f"Could not load image at {args.image_path}")

    result_image, figure_bboxes = analyze_image(original_img, predictor, args.confidence)

    # Save results
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved to {output_path}")
    print(f"\nDetected figures with confidence > {args.confidence:.1f}:")
    print(f"Original image dimensions: {original_img.shape[1]}x{original_img.shape[0]}")
    for i, figure in enumerate(figure_bboxes):
        print(f"Figure {i+1}: Confidence: {figure['confidence']:.2f}, BBox: [x1={figure['bbox'][0]:.1f}, y1={figure['bbox'][1]:.1f}, x2={figure['bbox'][2]:.1f}, y2={figure['bbox'][3]:.1f}]")