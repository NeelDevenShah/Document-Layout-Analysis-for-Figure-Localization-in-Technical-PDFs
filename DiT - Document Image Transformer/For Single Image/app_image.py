import os
import sys
import subprocess
from pathlib import Path
import cv2
import torch
from huggingface_hub import hf_hub_download
from concurrent.futures import ThreadPoolExecutor

def setup_dependencies():
    """Setup dependencies using subprocess and parallel downloads"""
    def clone_repo(repo_url, repo_name):
        if not Path(repo_name).exists():
            subprocess.run(['git', 'clone', repo_url], check=True)
            
    def setup_unilm():
        if not Path('unilm').exists():
            subprocess.run(['git', 'clone', 'https://github.com/microsoft/unilm.git'], check=True)
            # Use pathlib for file modification instead of sed
            file_path = Path('unilm/dit/object_detection/ditod/table_evaluation/data_structure.py')
            content = file_path.read_text()
            content = content.replace(
                'from collections import Iterable',
                'from collections.abc import Iterable'
            )
            file_path.write_text(content)

    def download_weights():
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
    
    # Install detectron2 after cloning, comment it out after first run
    if Path('detectron2').exists():
        subprocess.run(['pip', 'install', '-e', 'detectron2'], check=True)

def initialize_model():
    """Initialize the model with caching"""
    sys.path.extend(['unilm', 'detectron2'])
    
    from unilm.dit.object_detection.ditod import add_vit_config
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file("cascade_dit_base.yml")
    
    # Cache the model weights
    filepath = hf_hub_download(
        repo_id="Sebas6k/DiT_weights",
        filename="publaynet_dit-b_cascade.pth",
        repo_type="model"
    )
    cfg.MODEL.WEIGHTS = filepath
    
    # Use mixed precision training if available
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cuda"
        cfg.MODEL.FP16_ENABLED = True
    else:
        cfg.MODEL.DEVICE = "cpu"

    return DefaultPredictor(cfg), cfg

class ImageAnalyzer:
    def __init__(self):
        setup_dependencies()
        self.predictor, self.cfg = initialize_model()
        
    @torch.cuda.amp.autocast()  # Enable automatic mixed precision
    def analyze_image(self, img, score_threshold=0.7):
        """Analyze image with optimized processing"""
        from detectron2.utils.visualizer import ColorMode, Visualizer
        from detectron2.data import MetadataCatalog

        md = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        md.set(thing_classes=["text", "title", "list", "table", "figure"] 
               if self.cfg.DATASETS.TEST[0] != 'icdar2019_test' else ["table"])

        # Run prediction with CUDA optimization if available
        with torch.no_grad():  # Disable gradient calculation
            output = self.predictor(img)["instances"]

        # Efficient filtering using tensor operations
        high_scores = output.scores > score_threshold
        output = output[high_scores]

        # Optimize visualization
        v = Visualizer(
            img[:, :, ::-1],
            md,
            scale=1.0,
            instance_mode=ColorMode.SEGMENTATION
        )
        result = v.draw_instance_predictions(output.to("cpu"))
        result_image = result.get_image()[:, :, ::-1]

        # Efficient bbox extraction using tensor operations
        figure_class_idx = md.thing_classes.index("figure")
        figure_mask = output.pred_classes == figure_class_idx
        figure_bboxes = output.pred_boxes[figure_mask].tensor.cpu().numpy().tolist()

        return result_image, figure_bboxes

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process an image and return the raw output.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--score_threshold', type=float, default=0.7, 
                      help='Confidence score threshold for predictions')
    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ImageAnalyzer()

    # Load and process image
    img = cv2.imread(args.image_path)
    if img is None:
        raise ValueError(f"Could not load image at {args.image_path}")

    # Process image
    result_image, output = analyzer.analyze_image(img, args.score_threshold)

    # Save results
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"Bounding boxes: {output}")
    print(f"Processed image saved to {output_path}")

if __name__ == '__main__':
    main()