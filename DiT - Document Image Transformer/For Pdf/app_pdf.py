import os
import sys
import json
import cv2
import pymupdf
import torch
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import subprocess
from huggingface_hub import hf_hub_download
from typing import List, Dict, Tuple

def setup_dependencies():
    """Setup dependencies using subprocess and parallel downloads"""
    def clone_repo(repo_url: str, repo_name: str) -> None:
        if not Path(repo_name).exists():
            subprocess.run(['git', 'clone', repo_url], check=True)
            
    def setup_unilm() -> None:
        if not Path('unilm').exists():
            subprocess.run(['git', 'clone', 'https://github.com/microsoft/unilm.git'], check=True)
            file_path = Path('unilm/dit/object_detection/ditod/table_evaluation/data_structure.py')
            content = file_path.read_text()
            content = content.replace(
                'from collections import Iterable',
                'from collections.abc import Iterable'
            )
            file_path.write_text(content)

    def download_weights() -> None:
        if not Path('publaynet_dit-b_cascade.pth').exists():
            url = "https://layoutlm.blob.core.windows.net/dit/dit-fts/publaynet_dit-b_cascade.pth"
            params = "?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D"
            subprocess.run(['curl', '-LJ', '-o', 'publaynet_dit-b_cascade.pth', url + params], check=True)

    # Run setup tasks in parallel
    with ThreadPoolExecutor() as executor:
        tasks = [
            executor.submit(clone_repo, 'https://github.com/facebookresearch/detectron2.git', 'detectron2'),
            executor.submit(setup_unilm),
            executor.submit(download_weights)
        ]
        for task in tasks:
            task.result()

    # Install detectron2 after cloning
    if Path('detectron2').exists():
        subprocess.run(['pip', 'install', '-e', 'detectron2'], check=True)

class PDFProcessor:
    def __init__(self, score_threshold: float = 0.7):
        self.score_threshold = score_threshold
        setup_dependencies()
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the detection model"""
        sys.path.extend(['unilm', 'detectron2'])
        
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from unilm.dit.object_detection.ditod import add_vit_config

        cfg = get_cfg()
        add_vit_config(cfg)
        cfg.merge_from_file("cascade_dit_base.yml")
        
        # Cache the model weights
        cfg.MODEL.WEIGHTS = hf_hub_download(
            repo_id="Sebas6k/DiT_weights",
            filename="publaynet_dit-b_cascade.pth",
            repo_type="model"
        )
        
        # Enable mixed precision if CUDA is available
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cuda"
            cfg.MODEL.FP16_ENABLED = True
        else:
            cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    @torch.cuda.amp.autocast()
    def analyze_image(self, img: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
        """Analyze image and return visualization with all objects and filtered figure boxes"""
        from detectron2.utils.visualizer import ColorMode, Visualizer
        from detectron2.data import MetadataCatalog

        md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        if self.cfg.DATASETS.TEST[0] == 'icdar2019_test':
            md.set(thing_classes=["table"])
        else:
            md.set(thing_classes=["text", "title", "list", "table", "figure"])

        with torch.no_grad():
            output = self.predictor(img)["instances"]

        # Visualize all detected objects (no filtering)
        v = Visualizer(
            img[:, :, ::-1],
            md,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION
        )
        result = v.draw_instance_predictions(output.to("cpu"))
        result_image = result.get_image()[:, :, ::-1]

        # Filter only figures with high confidence for saving
        figure_class_idx = md.thing_classes.index("figure")
        figure_bboxes = []
        
        for i in range(len(output)):
            if (output.pred_classes[i] == figure_class_idx and 
                output.scores[i] > self.score_threshold):
                bbox = output.pred_boxes[i].tensor.cpu().numpy().tolist()[0]
                figure_bboxes.append(bbox)

        return result_image, figure_bboxes

    def process_pdf(self, pdf_path: str) -> None:
        """Process PDF and extract figures with improved error handling"""
        doc = pymupdf.open(pdf_path)
        pdf_name = Path(pdf_path).stem
        output_dir = Path.cwd() / pdf_name
        figures_dir = output_dir / "figures"
        
        output_dir.mkdir(exist_ok=True)
        figures_dir.mkdir(exist_ok=True)

        figures_data = []

        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = np.frombuffer(pix.tobytes(), np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                if img is None:
                    print(f"Warning: Could not load page {page_num + 1}")
                    continue

                result_image, figure_bboxes = self.analyze_image(img)

                # Save result image (shows all detected objects)
                result_path = output_dir / f"page_{page_num + 1}.jpg"
                cv2.imwrite(str(result_path), result_image)

                # Process only high-confidence figures
                for i, bbox in enumerate(figure_bboxes):
                    try:
                        x_min, y_min, x_max, y_max = map(int, bbox)

                        # Validate bbox dimensions
                        if x_min >= x_max or y_min >= y_max:
                            print(f"Warning: Invalid bbox dimensions on page {page_num + 1}, figure {i + 1}")
                            continue

                        # Extract figure
                        figure_img = img[y_min:y_max, x_min:x_max]
                        if figure_img.size == 0:
                            print(f"Warning: Empty figure on page {page_num + 1}, figure {i + 1}")
                            continue

                        # Save figure
                        figure_path = figures_dir / f"page_{page_num + 1}_figure_{i + 1}.jpg"
                        cv2.imwrite(str(figure_path), figure_img)

                        figures_data.append({
                            "page": page_num + 1,
                            "figure_index": i + 1,
                            "coordinates": bbox,
                            "path": str(figure_path)
                        })

                    except Exception as e:
                        print(f"Error processing figure {i + 1} on page {page_num + 1}: {str(e)}")

            except Exception as e:
                print(f"Error processing page {page_num + 1}: {str(e)}")

        # Save metadata
        json_path = output_dir / f"{pdf_name}_figures.json"
        with open(json_path, 'w') as f:
            json.dump(figures_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process a PDF and extract figures.')
    parser.add_argument('pdf_path', type=str, help='Path to the input PDF')
    parser.add_argument('--score_threshold', type=float, default=0.7,
                      help='Confidence score threshold for figure detection')
    args = parser.parse_args()

    processor = PDFProcessor(score_threshold=args.score_threshold)
    processor.process_pdf(args.pdf_path)
    print(f"Processing complete for {args.pdf_path}")

if __name__ == '__main__':
    main()