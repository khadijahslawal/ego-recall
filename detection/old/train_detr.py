"""
EgoRecall — Deformable DETR Training Script (GCP VM)
Run after setup_yolo_training.sh completes (data already extracted).
Uses a stratified subset of clips to keep training time manageable.

Usage: python3 train_detr.py
Estimated time: ~8 hours on T4 GPU (20 epochs, 200 train clips)
"""

import json
import os
import random
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from google.cloud import storage
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
from PIL import Image

# ── Config ─────────────────────────────────────────────────────────────────
BUCKET_NAME   = "egorecall-data"
DATA_DIR      = Path.home() / "egorecall_detection"  # already extracted by setup script
RESULTS_DIR   = Path.home() / "results" / "detr"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETR_MODEL    = "SenseTime/deformable-detr"
NUM_CLASSES   = 1
BATCH_SIZE    = 4
EPOCHS        = 20
LR            = 1e-5
RANDOM_SEED   = 42

# Subset config — full dataset is too large for DETR per epoch
N_TRAIN_CLIPS = 200   # ~200 clips × ~90 frames = ~18K train frames
N_VAL_CLIPS   = 50    # ~50 clips × ~90 frames = ~4.5K val frames

CHECKPOINT_PATH = RESULTS_DIR / "detr_checkpoint.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ── GCS client ─────────────────────────────────────────────────────────────
gcs_client = storage.Client()
bucket     = gcs_client.bucket(BUCKET_NAME)
print(f"Connected to gs://{BUCKET_NAME}")

# ── Step 1: Build stratified subset ───────────────────────────────────────
print("\n[1/5] Building stratified clip subset...")

# Check if subset already defined (for reproducibility across runs)
SUBSET_GCS_PATH = "processed/detr_subset_clips.json"
try:
    subset_data = json.loads(
        bucket.blob(SUBSET_GCS_PATH).download_as_bytes()
    )
    subset_train_clips = subset_data["train_clips"]
    subset_val_clips   = subset_data["val_clips"]
    print(f"  Loaded existing subset from GCS.")
except Exception:
    # Build new subset from available clip directories
    random.seed(RANDOM_SEED)
    train_img_base = DATA_DIR / "train" / "images"
    val_img_base   = DATA_DIR / "val"   / "images"

    all_train_clips = [d.name for d in train_img_base.iterdir() if d.is_dir()]
    all_val_clips   = [d.name for d in val_img_base.iterdir()   if d.is_dir()]

    subset_train_clips = random.sample(
        all_train_clips, min(N_TRAIN_CLIPS, len(all_train_clips))
    )
    subset_val_clips = random.sample(
        all_val_clips, min(N_VAL_CLIPS, len(all_val_clips))
    )

    # Save to GCS for reproducibility
    subset_data = {
        "train_clips": subset_train_clips,
        "val_clips"  : subset_val_clips,
        "random_seed": RANDOM_SEED,
    }
    bucket.blob(SUBSET_GCS_PATH).upload_from_string(
        json.dumps(subset_data)
    )
    print(f"  Built new subset and saved to GCS.")

print(f"  Train clips : {len(subset_train_clips)}")
print(f"  Val clips   : {len(subset_val_clips)}")

# ── Step 2: Verify subset data exists locally ──────────────────────────────
print("\n[2/5] Verifying local data...")

train_img_dir = DATA_DIR / "train" / "images"
train_lbl_dir = DATA_DIR / "train" / "labels"
val_img_dir   = DATA_DIR / "val"   / "images"
val_lbl_dir   = DATA_DIR / "val"   / "labels"

# Count frames only in subset clips
train_frames = sum(
    len(list((train_img_dir / cuid).glob("*.jpg")))
    for cuid in subset_train_clips
    if (train_img_dir / cuid).exists()
)
val_frames = sum(
    len(list((val_img_dir / cuid).glob("*.jpg")))
    for cuid in subset_val_clips
    if (val_img_dir / cuid).exists()
)
print(f"  Train frames in subset : {train_frames:,}")
print(f"  Val frames in subset   : {val_frames:,}")

# ── Step 3: Dataset and DataLoader ────────────────────────────────────────
print("\n[3/5] Building dataset...")

class EgoRecallDataset(Dataset):
    """
    Reads frames from a list of clips.
    Converts YOLO-format labels to DETR format on the fly.
    """
    def __init__(self, img_dir, lbl_dir, clip_list, processor):
        self.samples   = []
        self.processor = processor

        for clip_uid in clip_list:
            clip_img = img_dir / clip_uid
            clip_lbl = lbl_dir / clip_uid
            if not clip_img.exists():
                continue
            for img_path in sorted(clip_img.glob("*.jpg")):
                lbl_path = clip_lbl / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    self.samples.append((img_path, lbl_path))

        print(f"  Dataset: {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img  = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        for line in lbl_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append([cx, cy, bw, bh])
            labels.append(0)

        # Handle empty frames (no labeled objects)
        if not boxes:
            boxes  = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]

        target = {
            "boxes"    : torch.tensor(boxes,  dtype=torch.float32),
            "labels"   : torch.tensor(labels, dtype=torch.long),
            "image_id" : torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
        }
        encoding     = self.processor(
            images=img, annotations=target, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze(0)
        target       = encoding["labels"][0]
        return pixel_values, target


def collate_fn(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


print("  Loading processor and model...")
detr_processor = AutoImageProcessor.from_pretrained(DETR_MODEL)

train_dataset = EgoRecallDataset(
    train_img_dir, train_lbl_dir, subset_train_clips, detr_processor
)
val_dataset = EgoRecallDataset(
    val_img_dir, val_lbl_dir, subset_val_clips, detr_processor
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate_fn,
    num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate_fn,
    num_workers=4, pin_memory=True
)

print(f"  Train batches : {len(train_loader):,}")
print(f"  Val batches   : {len(val_loader):,}")

# ── Step 4: Train ──────────────────────────────────────────────────────────
print("\n[4/5] Loading Deformable DETR for fine-tuning...")

detr_model = DeformableDetrForObjectDetection.from_pretrained(
    DETR_MODEL,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
detr_model = detr_model.to(DEVICE)

optimizer = AdamW(detr_model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-4,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)

train_losses  = []
val_losses    = []
best_val_loss = float("inf")
best_path     = RESULTS_DIR / "detr_best.pt"
last_path     = RESULTS_DIR / "detr_last.pt"

# Load checkpoint if resuming
start_epoch = 0
if CHECKPOINT_PATH.exists():
    ckpt = json.loads(CHECKPOINT_PATH.read_text())
    start_epoch  = ckpt["epoch"]
    train_losses = ckpt["train_losses"]
    val_losses   = ckpt["val_losses"]
    best_val_loss = ckpt["best_val_loss"]
    if last_path.exists():
        detr_model.load_state_dict(torch.load(last_path))
        print(f"  Resumed from epoch {start_epoch}")

print(f"\nStarting training from epoch {start_epoch + 1}/{EPOCHS}...")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  LR         : {LR}")
print(f"  Device     : {DEVICE}")

for epoch in range(start_epoch, EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────
    detr_model.train()
    epoch_loss = 0.0

    for pixel_values, targets in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"
    ):
        pixel_values = pixel_values.to(DEVICE)
        targets      = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        outputs = detr_model(pixel_values=pixel_values, labels=targets)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(detr_model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        epoch_loss += outputs.loss.item()

    avg_train = epoch_loss / len(train_loader)
    train_losses.append(avg_train)

    # ── Validate ───────────────────────────────────────────────────────────
    detr_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixel_values, targets in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]  "
        ):
            pixel_values = pixel_values.to(DEVICE)
            targets      = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            val_loss    += detr_model(pixel_values=pixel_values, labels=targets).loss.item()

    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train loss: {avg_train:.4f} | "
        f"Val loss: {avg_val:.4f}"
    )

    # Save last checkpoint
    torch.save(detr_model.state_dict(), last_path)

    # Save best
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(detr_model.state_dict(), best_path)
        print(f"  → Best saved (val_loss={best_val_loss:.4f})")

    # Save checkpoint every epoch for safe resume
    CHECKPOINT_PATH.write_text(json.dumps({
        "epoch"        : epoch + 1,
        "train_losses" : train_losses,
        "val_losses"   : val_losses,
        "best_val_loss": best_val_loss,
    }))

    # Upload to GCS every 5 epochs
    if (epoch + 1) % 5 == 0:
        bucket.blob("models/detr_finetuned/last.pt").upload_from_filename(str(last_path))
        print(f"  → Checkpoint uploaded to GCS (epoch {epoch+1})")

# ── Step 5: Upload final weights and metrics ──────────────────────────────
print("\n[5/5] Uploading to GCS...")

if best_path.exists():
    bucket.blob("models/detr_finetuned/best.pt").upload_from_filename(str(best_path))
    print(f"  Uploaded: gs://{BUCKET_NAME}/models/detr_finetuned/best.pt")

if last_path.exists():
    bucket.blob("models/detr_finetuned/last.pt").upload_from_filename(str(last_path))
    print(f"  Uploaded: gs://{BUCKET_NAME}/models/detr_finetuned/last.pt")

# Save training metrics
metrics_df = pd.DataFrame({
    "epoch"      : list(range(1, len(train_losses) + 1)),
    "train_loss" : train_losses,
    "val_loss"   : val_losses,
})
metrics_df.to_csv(str(RESULTS_DIR / "detr_results.csv"), index=False)
bucket.blob("models/detr_finetuned/results.csv").upload_from_filename(
    str(RESULTS_DIR / "detr_results.csv")
)
print(f"  Uploaded: gs://{BUCKET_NAME}/models/detr_finetuned/results.csv")

print("\n── Training complete ────────────────────────────────")
print(f"  Epochs         : {len(train_losses)}")
print(f"  Best val loss  : {best_val_loss:.4f}")
print(f"  Train clips    : {len(subset_train_clips)} ({train_frames:,} frames)")
print(f"  Val clips      : {len(subset_val_clips)} ({val_frames:,} frames)")
print("────────────────────────────────────────────────────")
print("\nRun 03b_detr.ipynb in Colab to evaluate and visualize results.")
