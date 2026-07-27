# MuseumSCAT — Specimen Collection Annotation Task

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task)
[![Workshop](https://img.shields.io/badge/ECCV-CVNH%202026-blue)](https://computer-vision-for-natural-heritage.github.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

OCR and text-type identification on museum specimen label photographs from the
**Danish Dung Beetle collection**, Natural History Museum of Denmark — built for the
**MuseumSCAT2026** Kaggle challenge, part of the **CVNH ECCV 2026 Workshop**.

<p align="center">
  <img src="https://computer-vision-for-natural-heritage.github.io/assets/images/museumscat_example.jpg" width="500" alt="Museum label example">
</p>

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Usage](#usage)
- [Approach](#approach)
- [Repo Structure](#repo-structure)
- [Results](#results)
- [References](#references)

## Overview

| | |
|---|---|
| **Competition** | [MuseumSCAT2026 on Kaggle](https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task) |
| **Workshop** | [CVNH @ ECCV 2026](https://computer-vision-for-natural-heritage.github.io/) — Sept 8, 2026 |
| **Deadline** | September 1, 2026 |
| **Task** | Text recognition (OCR) + text-type classification (`date`, `locality`, `collector`, `species`, etc.) on museum label images |
| **Dataset size** | ~3,500 images with annotations |

## Dataset

Exact file layout, columns, and submission CSV spec are on Kaggle's **Data** /
**Evaluation** tabs — confirm there before writing the pipeline.

```bash
kaggle competitions download -c museumscat-specimen-collection-annotation-task
unzip museumscat-specimen-collection-annotation-task.zip -d data/
```

Expect: handwritten/typed text, fading, rotation, mixed Danish/Latin vocabulary.

## Setup

```bash
git clone <this-repo>
cd museumscat
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`
```
torch
torchvision
transformers
pandas
numpy
opencv-python
pillow
kaggle
```

## Usage

```bash
# Preprocess (deskew, crop, contrast enhance)
python src/preprocessing.py --input data/raw --output data/processed

# Train
python src/train.py --config configs/baseline.yaml

# Predict + generate submission
python src/predict.py --checkpoint checkpoints/best.pt --output submissions/submission.csv
```

## Approach

1. **Baseline** — off-the-shelf OCR to validate the pipeline end-to-end:
   [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) · [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) · [EasyOCR](https://github.com/JaidedAI/EasyOCR)
2. **Fine-tuning** — adapt OCR on the provided training labels.
3. **Text-type classification** — small transformer / CRF over recognized spans,
   or a layout-aware model (e.g. LayoutLM) if bounding boxes are provided.
4. **Preprocessing** — deskew, contrast enhancement, crop-to-label.

## Repo Structure

```
.
├── data/                # raw + processed dataset (gitignored)
├── notebooks/           # EDA, prototyping
├── configs/              # training configs
├── src/
│   ├── preprocessing.py
│   ├── ocr_model.py
│   ├── classifier.py
│   ├── train.py
│   └── predict.py
├── submissions/          # generated CSVs
├── requirements.txt
└── README.md
```

## Results

| Model | Metric | Score |
|---|---|---|
| TBA | TBA | TBA |

## References

- Competition: https://www.kaggle.com/competitions/museumscat-specimen-collection-annotation-task
- Workshop: https://computer-vision-for-natural-heritage.github.io/
- Organizers: DTU, KU, SNM (Natural History Museum of Denmark), QIM

---
Eligibility note: a 1-page method PDF + 1 slide are required (not for entry, but to
qualify as a challenge winner / paper co-author).
