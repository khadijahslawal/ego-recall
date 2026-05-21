"""
EgoRecall — YOLOv8 Training Script (GCP VM)
Run after setup_yolo_training.sh completes.
Usage: python3 train_yolo.py
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from google.cloud import storage
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET_NAME = "egorecall-data"
DATA_DIR    = Path.home() / "egorecall_detection"
RESULTS_DIR = Path.home() / "results" / "detection"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
YOLO_MODEL    = "yolov8s.pt"
IMG_SIZE      = 320        # balanced speed vs accuracy for small objects
EPOCHS        = 20         # more epochs since VM won't timeout
BATCH_SIZE    = 16
DEVICE        = "0"        # GPU 0; use "cpu" if no GPU

# ── GCS client ─────────────────────────────────────────────────────────────
gcs_client = storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)
print(f"Connected to gs://{BUCKET_NAME}")

# ── Step 1: Verify data structure ──────────────────────────────────────────
print("\n[1/4] Verifying data structure...")

# After setup_yolo_training.sh with --strip-components=3:
# DATA_DIR/train/images/{clip_uid}/{frame}.jpg  ← clean, no extra subdir
# DATA_DIR/train/labels/{clip_uid}/{frame}.txt

train_imgs = list((DATA_DIR / "train" / "images").rglob("*.jpg"))
val_imgs   = list((DATA_DIR / "val"   / "images").rglob("*.jpg"))
train_lbls = list((DATA_DIR / "train" / "labels").rglob("*.txt"))
val_lbls   = list((DATA_DIR / "val"   / "labels").rglob("*.txt"))

print(f"  Train images : {len(train_imgs):,}")
print(f"  Train labels : {len(train_lbls):,}")
print(f"  Val images   : {len(val_imgs):,}")
print(f"  Val labels   : {len(val_lbls):,}")

# Verify sample label
sample_img = train_imgs[0]
sample_lbl = (DATA_DIR / "train" / "labels" / sample_img.parent.name / sample_img.stem).with_suffix(".txt")
print(f"\n  Sample image : {sample_img}")
print(f"  Sample label : {sample_lbl}")
print(f"  Label exists : {sample_lbl.exists()}")
if sample_lbl.exists():
    print(f"  Label content: {sample_lbl.read_text().strip()[:80]}")

# ── Step 2: Flatten with hard links ────────────────────────────────────────
# YOLO requires labels in the same directory as images (or auto-mapped path)
# We put both in the same flat directory using hard links (no extra disk space)
print("\n[2/4] Flattening dataset structure...")

def flatten_split(split):
    img_dir  = DATA_DIR / split / "images"
    lbl_dir  = DATA_DIR / split / "labels"
    flat_dir = DATA_DIR / split / f"images_{split}_flat"
    flat_dir.mkdir(exist_ok=True)

    imgs = list(img_dir.rglob("*.jpg"))
    n_linked = 0

    for img_path in tqdm(imgs, desc=f"  {split}"):
        clip_uid = img_path.parent.name
        new_stem = f"{clip_uid}_{img_path.stem}"

        # Hard link image
        dst_img = flat_dir / f"{new_stem}.jpg"
        if not dst_img.exists():
            os.link(img_path, dst_img)

        # Hard link label into same directory
        lbl_path = lbl_dir / clip_uid / f"{img_path.stem}.txt"
        dst_lbl  = flat_dir / f"{new_stem}.txt"
        if lbl_path.exists() and not dst_lbl.exists():
            os.link(lbl_path, dst_lbl)
            n_linked += 1

    flat_imgs = list(flat_dir.glob("*.jpg"))
    flat_lbls = list(flat_dir.glob("*.txt"))
    print(f"  {split}: {len(flat_imgs):,} images, {len(flat_lbls):,} labels")
    return len(flat_imgs)

n_train = flatten_split("train")
n_val   = flatten_split("val")
print(f"  Total: {n_train + n_val:,} frames")

# ── Step 3: Write dataset.yaml ─────────────────────────────────────────────
print("\n[3/4] Writing dataset.yaml...")
dataset_yaml = DATA_DIR / "dataset.yaml"
yaml_content = f"""path: {DATA_DIR}
train: train/images_train_flat
val: val/images_val_flat

nc: 1
names: ['object']
"""
dataset_yaml.write_text(yaml_content)
print(dataset_yaml.read_text())

# Verify YOLO can find labels — images_train_flat → labels_train_flat
# But we put labels IN the images_train_flat directory
# So YOLO will find them directly without path substitution
flat_lbls = list((DATA_DIR / "train" / "images_train_flat").glob("*.txt"))
print(f"  Labels in flat dir: {len(flat_lbls):,} (should match train count)")

# ── Step 4: Train YOLOv8 ───────────────────────────────────────────────────
print("\n[4/4] Starting YOLOv8 training...")
print(f"  Model      : {YOLO_MODEL}")
print(f"  Image size : {IMG_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Device     : {DEVICE}")

model = YOLO(YOLO_MODEL)

train_results = model.train(
    data          = str(dataset_yaml),
    epochs        = EPOCHS,
    imgsz         = IMG_SIZE,
    batch         = BATCH_SIZE,
    device        = DEVICE,
    project       = str(RESULTS_DIR),
    name          = "yolov8s_320",
    lr0           = 0.01,
    lrf           = 0.01,
    warmup_epochs = 3,
    # Augmentation
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    flipud        = 0.0,   # egocentric video has consistent orientation
    fliplr        = 0.5,
    mosaic        = 1.0,
    plots         = True,
    verbose       = True,
    save_period   = 5,     # save checkpoint every 5 epochs
)

# ── Upload weights and results to GCS ─────────────────────────────────────
print("\nUploading weights to GCS...")
run_dir   = RESULTS_DIR / "yolov8s_320"
best_pt   = run_dir / "weights" / "best.pt"
last_pt   = run_dir / "weights" / "last.pt"

if best_pt.exists():
    bucket.blob("models/yolov8s_finetuned/best.pt").upload_from_filename(str(best_pt))
    print(f"  Uploaded: gs://{BUCKET_NAME}/models/yolov8s_finetuned/best.pt")

if last_pt.exists():
    bucket.blob("models/yolov8s_finetuned/last.pt").upload_from_filename(str(last_pt))
    print(f"  Uploaded: gs://{BUCKET_NAME}/models/yolov8s_finetuned/last.pt")

# Upload training results CSV
results_csv = run_dir / "results.csv"
if results_csv.exists():
    bucket.blob("models/yolov8s_finetuned/results.csv").upload_from_filename(str(results_csv))
    print(f"  Uploaded: gs://{BUCKET_NAME}/models/yolov8s_finetuned/results.csv")

    # Print final metrics
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    print("\n── Training complete ──────────────────────────────────")
    print(f"  Epochs trained : {len(df)}")
    if "metrics/mAP50(B)" in df.columns:
        print(f"  Best mAP@0.5   : {df['metrics/mAP50(B)'].max():.4f}")
    if "metrics/mAP50-95(B)" in df.columns:
        print(f"  Best mAP@0.5:95: {df['metrics/mAP50-95(B)'].max():.4f}")
    print("────────────────────────────────────────────────────────")

print("\nDone. Run 03a_yolov8_eval.ipynb in Colab to evaluate and visualize results.")
