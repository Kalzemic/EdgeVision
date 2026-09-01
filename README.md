# EdgeVision

**A SAM3-distilled YOLOv8s object detector for edge devices.**

EdgeVision is a teacher–student computer vision pipeline that produces a lightweight person / vehicle detector deployable on a Jetson Orin Nano via TensorRT. It was built as the capstone project for a Computer Vision course, and demonstrates the full loop from automated dataset labeling through on-device inference.

The core idea: use a large, expensive foundation model (SAM3) as an offline "teacher" to generate high-quality bounding-box labels on an unlabeled dataset, then train a small, fast student (YOLOv8s) that can actually run at the edge.

---

## Pipeline overview

```
Flickr30k captions ─► generate_prompts.py ─► prompts.csv
                                                │
                                                ▼
Flickr30k images ─► generate_labels.py (SAM3 as teacher) ─► YOLO-format labels
                                                │
                                                ▼
                                        data_split.py
                                                │
                                                ▼
                              train.py (YOLOv8s student, DetectorLoss)
                                                │
                                                ▼
                                    Export to ONNX → TensorRT
                                                │
                                                ▼
                                Deploy on Jetson Orin Nano
```

## Stages

### 1. Prompt generation (`generate_prompts.py`)
Parses the Flickr30k caption CSV and regex-matches each image against curated keyword sets for the two target classes (persons and vehicles). Produces a `prompts.csv` mapping each image to the class prompts SAM3 should be conditioned on. Aggregates over multiple captions per image (any caption mentioning the class is enough to include it).

### 2. Auto-labeling (`generate_labels.py`)
Feeds each image and its prompts through SAM3 to obtain bounding boxes for the target classes, then writes labels in YOLO format alongside the images.

### 3. Data split (`data_split.py`)
Partitions the labeled set into train / val / test splits, writing `data.yaml` (main config) and `overfit.yaml` (small subset for sanity-checking the training loop).

### 4. Training (`train.py`, `DetectorLoss.py`)
Trains YOLOv8s on the SAM3-generated labels. `DetectorLoss.py` contains the loss configuration used during training. Augmentation was tuned against the specific failure modes observed in baseline predictions (small and crowded objects in particular).

### 5. Verification (`label_test.py`, `file_test.py`, `check.jpg`)
Quick smoke tests: visualize generated labels on random samples, verify file layout, sanity-check a single image.

### 6. Deployment (not in this repo)
Trained weights are exported to ONNX and then to a TensorRT engine, and served from a Python inference script on a Jetson Orin Nano.

---

## Repo layout

```
EdgeVision/
├── EdgeVision.ipynb      # Main notebook: end-to-end walk-through
├── generate_prompts.py   # Stage 1: caption → class prompts
├── generate_labels.py    # Stage 2: SAM3 auto-labeling
├── data_split.py         # Stage 3: train/val split, config generation
├── train.py              # Stage 4: YOLOv8s training entry point
├── DetectorLoss.py       # Loss configuration used in training
├── data.yaml             # Full training config
├── overfit.yaml          # Small-subset config for overfitting sanity checks
├── label_test.py         # Label visualization / spot-check utility
├── file_test.py          # Filesystem layout sanity check
├── check.jpg             # Sample image for quick verification
├── cleanup.sh            # Reset intermediate artifacts
├── requirements.txt      # Python dependencies
└── experiments/          # Training runs, logs, weights
```

---

## Stack

- **Foundation model (teacher):** SAM3
- **Student:** YOLOv8s (Ultralytics)
- **Dataset:** Flickr30k
- **Deployment target:** NVIDIA Jetson Orin Nano
- **Inference runtime:** TensorRT (via ONNX)
- **Language:** Python (PyTorch)

## Reproducing

```bash
pip install -r requirements.txt

python generate_prompts.py     # build prompts.csv
python generate_labels.py      # run SAM3 to produce labels
python data_split.py           # split + write data.yaml
python train.py                # train YOLOv8s
```

Weights and training artifacts land in `experiments/`.

---

## Notes

- Class set is intentionally narrow (person, vehicle) to keep the student small and the deployment target realistic for embedded inference.
- The teacher–student setup means the pipeline generalizes: swap the prompt-generation stage and you can retarget the detector to any classes SAM3 can be prompted for, without any hand-annotated data.
