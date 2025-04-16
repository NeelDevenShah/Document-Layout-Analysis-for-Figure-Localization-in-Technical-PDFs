# Document Layout Analysis for Figure Localization in Technical PDFs

This repository provides three state-of-the-art models fine-tuned on the PubLayNet dataset for precise figure localization in technical PDFs, including AI conference papers, financial reports (BSE/NASDAQ), and scientific documents. The models address the challenge of accurately extracting bounding boxes for figures, flowcharts, and diagrams from complex PDF layouts, where traditional methods (e.g., fitz, pdfplumber, Sobel edge detection) and zero-shot vision APIs (GPT-4, Claude) fail due to fragmented outputs or misaligned coordinates.

---

## 📌 Problem Statement

Technical PDFs often use LaTeX-generated layouts, causing conventional PDF parsers to:

- Extract partial image fragments instead of complete figures.
- Fail to aggregate bounding boxes for multi-component diagrams.
- Misalign coordinates for flowcharts or annotated visuals.

### Why Deep Learning?

PubLayNet’s 360k+ annotated document images enable models to learn semantic layout patterns (text, titles, lists, tables, figures) and generalize to unseen PDF structures. Our fine-tuned models outperform rule-based and zero-shot approaches by leveraging spatial context and hierarchical relationships.

---

## 🗃️ Dataset: PubLayNet

- **Source**: Automatically parsed from 1M+ PubMed Central PDFs.
- **Annotations**: Bounding boxes for text, title, list, table, and figure.
- **Split**: 335k training / 11k validation images.
- **Key Advantage**: Large-scale, domain-agnostic layout annotations ideal for transfer learning.
- **Usage in this project**: The dataset was divided into 10 parts, and we used one part to fine-tune the model for figure localization.

---

## 🧠 Models

### 1. Document Image Transformer (DiT)

- **Architecture**: Mask R-CNN with ViT backbone, pre-trained on 42M document images via self-supervised learning.
- **Fine-tuning**: Cascade Mask R-CNN on PubLayNet for enhanced layout reasoning.

### 2. YOLOv11 Ultralytics

- **Architecture**: Nano (11n) to X-Large (11x) variants; optimized for real-time inference.
- **Fine-tuning**: Trained on PubLayNet with 640px resolution and figure-centric augmentations.

### 3. Detectron2 (Mask R-CNN)

- **Backbones**: ResNet-50/101, ResNeXt-101-32x8d.
- **Usage**: Supports CPU/GPU inference and Docker deployment.

---

## 📊 Results & Workflow

- **PDF Preprocessing**: Convert PDF pages to images (600dpi recommended) using `pdf2image`.
- **Figures**: The repository contains visual outputs that demonstrate the results of figure localization.
- **Code Availability**: The repository provides all necessary scripts for fine-tuning and inference.

---

## 📮 Contact

For issues or collaborations, open a GitHub Issue or email [neeldevenshah.ai@gmail.com](mailto:neeldevenshah.ai@gmail.com).

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](./LICENSE) file for details.
