import os
import sys

if not os.path.exists('detectron2'):
    os.system('git clone https://github.com/facebookresearch/detectron2.git')
    os.system('pip install -e detectron2')

if not os.path.exists('unilm'):
    os.system("git clone https://github.com/microsoft/unilm.git")
    os.system("sed -i 's/from collections import Iterable/from collections.abc import Iterable/' unilm/dit/object_detection/ditod/table_evaluation/data_structure.py")

if not os.path.exists('publaynet_dit-b_cascade.pth'):
    os.system("curl -LJ -o publaynet_dit-b_cascade.pth 'https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D'")

sys.path.append("unilm")
sys.path.append("detectron2")

import cv2

from unilm.dit.object_detection.ditod import add_vit_config

import torch

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor

from huggingface_hub import hf_hub_download


# Step 1: instantiate config
cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("cascade_dit_base.yml")

# Step 2: add model weights URL to config
filepath = hf_hub_download(repo_id="Sebas6k/DiT_weights", filename="publaynet_dit-b_cascade.pth", repo_type="model")
cfg.MODEL.WEIGHTS = filepath

# Step 3: set device
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Step 4: define model
predictor = DefaultPredictor(cfg)


def analyze_image(img):
    md = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    if cfg.DATASETS.TEST[0] == 'icdar2019_test':
        md.set(thing_classes=["table"])
    else:
        md.set(thing_classes=["text", "title", "list", "table", "figure"])

    output = predictor(img)["instances"]
    
    # Filter predictions based on score
    high_score_idxs = [i for i, score in enumerate(output.scores) if score > 0.7]
    output = output[high_score_idxs]

    v = Visualizer(img[:, :, ::-1],
                   md,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # Extract bounding boxes for figures
    figure_bboxes = []
    for i in range(len(output)):
        if output.pred_classes[i] == md.thing_classes.index("figure"):
            bbox = output.pred_boxes[i].tensor.cpu().numpy().tolist()
            figure_bboxes.append(bbox)

    return result_image, figure_bboxes
    
if __name__ == '__main__':
    import argparse

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process an image and return the raw output.')
    parser.add_argument('image_path', type=str, help='Path to the input image')

    args = parser.parse_args()

    # Load the image
    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"Could not load image at {args.image_path}")

    # Process the image
    result_image, output = analyze_image(img)

    # Save or display the result
    output_path = "output_image.jpg"
    print(output)
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved to {output_path}")