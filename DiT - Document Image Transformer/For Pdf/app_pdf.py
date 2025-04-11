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

# TODO: Change(Increase) the threshold for picking out the figure from the pdf

import json
import cv2
import pymupdf
import torch
import argparse
import numpy as np
from huggingface_hub import hf_hub_download
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from unilm.dit.object_detection.ditod import add_vit_config

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
    v = Visualizer(img[:, :, ::-1],
                   md,
                   scale=1.0,
                   instance_mode=ColorMode.SEGMENTATION)
    result = v.draw_instance_predictions(output.to("cpu"))
    result_image = result.get_image()[:, :, ::-1]

    # Extract bounding boxes for figures with a score greater than 0.7
    figure_bboxes = []
    for i in range(len(output)):
        if output.pred_classes[i] == md.thing_classes.index("figure") and output.scores[i] > 0.7:
            bbox = output.pred_boxes[i].tensor.cpu().numpy().tolist()
            figure_bboxes.append(bbox)

    return result_image, figure_bboxes

def process_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(os.getcwd(), pdf_name)
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    figures_data = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            continue

        result_image, figure_bboxes = analyze_image(img)

        # Save the result image
        result_image_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
        cv2.imwrite(result_image_path, result_image)

        # Save figures and their coordinates
        for i, bbox in enumerate(figure_bboxes):
            print(f"Processing page {page_num + 1}, figure {i + 1}, bbox: {bbox}")

            # Ensure bbox is correctly formatted
            if len(bbox) == 1:  # If it's wrapped in another list, extract the first element
                bbox = bbox[0]

            if len(bbox) != 4:
                print(f"Skipping malformed bbox on page {page_num + 1}, figure {i + 1}: {bbox}")
                continue

            x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to integers

            # Ensure bbox is valid
            if x_min >= x_max or y_min >= y_max:
                print(f"Skipping invalid bbox on page {page_num + 1}, figure {i + 1} due to incorrect dimensions: {bbox}")
                continue

            # Crop the figure
            figure_img = img[y_min:y_max, x_min:x_max]

            if figure_img.size == 0:
                print(f"Empty image for bbox on page {page_num + 1}, figure {i + 1}. Skipping.")
                continue

            # Save cropped figure
            figure_path = os.path.join(figures_dir, f"page_{page_num + 1}_figure_{i + 1}.jpg")
            cv2.imwrite(figure_path, figure_img)
            figures_data.append({
                "page": page_num + 1,
                "figure_index": i + 1,
                "coordinates": bbox,
                "path": figure_path
            })



    # Save the figures data to a JSON file
    json_path = os.path.join(output_dir, f"{pdf_name}_figures.json")
    with open(json_path, 'w') as json_file:
        json.dump(figures_data, json_file, indent=4)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process a PDF and extract figures.')
    parser.add_argument('pdf_path', type=str, help='Path to the input PDF')

    args = parser.parse_args()

    # Process the PDF
    process_pdf(args.pdf_path)
    print(f"Processed PDF saved to {args.pdf_path}")