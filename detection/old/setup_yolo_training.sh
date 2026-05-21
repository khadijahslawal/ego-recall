#!/bin/bash
# EgoRecall — YOLOv8 Training Setup Script
# Run this on the GCP VM before launching train_yolo.py
# Usage: bash setup_yolo_training.sh

set -e  # exit on error

BUCKET="gs://egorecall-data"
DATA_DIR="$HOME/egorecall_detection"
TMP_DIR="$HOME/tmp_tars"

echo "=== EgoRecall YOLOv8 Training Setup ==="
echo "Data dir : $DATA_DIR"
echo "Bucket   : $BUCKET"
echo ""

# ── Install dependencies ───────────────────────────────────────────────────
echo "[1/5] Installing dependencies..."
pip3 install ultralytics google-cloud-storage tqdm pandas pyarrow --quiet
echo "Done."

# ── Create directories ─────────────────────────────────────────────────────
echo "[2/5] Creating directories..."
mkdir -p $DATA_DIR/train/images
mkdir -p $DATA_DIR/train/labels
mkdir -p $DATA_DIR/val/images
mkdir -p $DATA_DIR/val/labels
mkdir -p $TMP_DIR
echo "Done."

# ── Download tar files from GCS ────────────────────────────────────────────
echo "[3/5] Downloading tar files from GCS..."
echo "  Downloading det_train.tar (~15GB)..."
gsutil cp $BUCKET/processed/det_train.tar $TMP_DIR/det_train.tar

echo "  Downloading det_val.tar (~5GB)..."
gsutil cp $BUCKET/processed/det_val.tar $TMP_DIR/det_val.tar

echo "  Downloading lbl_train.tar..."
gsutil cp $BUCKET/processed/lbl_train.tar $TMP_DIR/lbl_train.tar

echo "  Downloading lbl_val.tar..."
gsutil cp $BUCKET/processed/lbl_val.tar $TMP_DIR/lbl_val.tar
echo "Done."

# ── Extract tars ───────────────────────────────────────────────────────────
# Use --strip-components=3 to remove home/username/det_train/ prefix
# leaving clean {clip_uid}/{frame}.jpg structure
echo "[4/5] Extracting tars..."

echo "  Extracting detection frames (train)..."
tar -xf $TMP_DIR/det_train.tar -C $DATA_DIR/train/images/ --strip-components=3

echo "  Extracting detection frames (val)..."
tar -xf $TMP_DIR/det_val.tar -C $DATA_DIR/val/images/ --strip-components=3

echo "  Extracting YOLO labels (train)..."
tar -xf $TMP_DIR/lbl_train.tar -C $DATA_DIR/train/labels/ --strip-components=3

echo "  Extracting YOLO labels (val)..."
tar -xf $TMP_DIR/lbl_val.tar -C $DATA_DIR/val/labels/ --strip-components=3

# ── Verify extraction ──────────────────────────────────────────────────────
echo ""
echo "Verifying extraction..."
TRAIN_IMGS=$(find $DATA_DIR/train/images -name "*.jpg" | wc -l)
VAL_IMGS=$(find $DATA_DIR/val/images -name "*.jpg" | wc -l)
TRAIN_LBLS=$(find $DATA_DIR/train/labels -name "*.txt" | wc -l)
VAL_LBLS=$(find $DATA_DIR/val/labels -name "*.txt" | wc -l)

echo "  Train images : $TRAIN_IMGS"
echo "  Train labels : $TRAIN_LBLS"
echo "  Val images   : $VAL_IMGS"
echo "  Val labels   : $VAL_LBLS"

if [ "$TRAIN_IMGS" -eq "$TRAIN_LBLS" ] && [ "$VAL_IMGS" -eq "$VAL_LBLS" ]; then
    echo "  Counts match."
else
    echo "  WARNING: image/label counts do not match."
fi

# ── Clean up tars to free disk space ──────────────────────────────────────
echo "[5/5] Cleaning up tar files..."
rm -rf $TMP_DIR
echo "Done."

echo ""
echo "=== Setup complete ==="
echo "Run training with: tmux new -s yolo_train && python3 train_yolo.py"
