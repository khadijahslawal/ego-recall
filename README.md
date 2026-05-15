# EgoRecall — Visual Memory Retrieval for Smart Glasses

**Course:** Applied Computer Vision · MS in Applied Data Science, University of Chicago  
**Dataset:** Ego4D Visual Object Queries (VQ) Benchmark  
**GCS Bucket:** `gs://egorecall-data`

---

## Project Overview

EgoRecall is a two-stage visual memory system that answers the question:  
*"Where did I last see this object?"* from egocentric smart glasses footage.

**Two CV problems:**

| Problem | Task | Models |
|---------|------|--------|
| 1 — Object Detection | Given a frame, detect the target object | YOLOv8s vs Deformable DETR |
| 2 — Visual Retrieval | Given a query image of an object, find the frame where it was last seen | CLIP+FAISS vs BLIP+FAISS |

---

## What Has Been Done

### Notebook 01 — Dataset Overview (`01_dataset_overview.ipynb`)
Full EDA on the Ego4D VQ annotations. Key findings:
- 18,114 valid query-sets across 1,727 unique videos
- 3,487 unique object titles — open vocabulary, long-tail distribution
- Median object normalized area: 1.9% of frame (small objects)
- All clips at 5 FPS, source videos at 30 FPS, 720×540 resolution
- Annotations drawn at 1440×1080 — 2× scale factor when reading actual frames

Outputs saved to GCS:
```
gs://egorecall-data/processed/vq_query_sets.parquet   ← one row per valid query-set
```

### Notebook 02 — Frame Extraction (`02_frame_extraction.ipynb`)
Extracted detection frames and visual crops from all 1,727 source videos.  
Ran as a Python script on the GCP VM overnight.

Outputs saved to GCS:
```
gs://egorecall-data/frames/detection/train/{clip_uid}/*.jpg   ← 184,332 frames
gs://egorecall-data/frames/detection/val/{clip_uid}/*.jpg     ← 61,874 frames
gs://egorecall-data/labels/yolo/train/{clip_uid}/*.txt        ← YOLO format labels
gs://egorecall-data/labels/yolo/val/{clip_uid}/*.txt
gs://egorecall-data/frames/retrieval/visual_crops/{split}/{clip_uid}/{annotation_uid}_{qs_id}.jpg ← 18,111 query images
gs://egorecall-data/processed/extraction_manifest.parquet
```

**Important:** Bounding box coordinates in annotations are in 1440×1080 space but frames are 720×540. YOLO labels are already normalized to actual frame resolution — no further scaling needed.

### Notebook 03 — Object Detection Baseline (`03_detection_baseline.ipynb`)
**Status: In progress (YOLOv8 training running)**

Zero-shot baseline (pretrained COCO weights, class-agnostic evaluation):
- YOLOv8s zero-shot mAP@0.5: **0.0125** (expected — domain mismatch)

Fine-tuning in progress on Colab T4 GPU.

### Notebook 04 — Visual Retrieval Baseline (`04_retrieval_baseline.ipynb`)
**Status: Ready to run**

Two versions available:
- `04_retrieval_baseline.ipynb` — Option A: pre-computed embeddings (full dataset, needs VM job first)
- `04_retrieval_baseline_subset.ipynb` — Option B: on-the-fly subset evaluation (500 clips, self-contained, no VM needed)

---

## GCS Bucket Structure

```
gs://egorecall-data/
│
├── ego4d/v2/
│   ├── annotations/
│   │   ├── vq_train.json          ← raw Ego4D annotations (1,321 videos)
│   │   └── vq_val.json            ← raw Ego4D annotations (422 videos)
│   └── video_540ss/
│       └── {video_uid}.mp4        ← 1,727 source videos (~400MB each, 720×540)
│
├── processed/
│   ├── vq_query_sets.parquet      ← flat table: one row per valid query-set
│   ├── extraction_manifest.parquet← frame extraction job summary
│   ├── detection_results.parquet  ← detection model results (after notebook 03)
│   ├── retrieval_results.parquet  ← retrieval model results (after notebook 04)
│   └── embed_index_frames.py      ← VM script for pre-computing index embeddings
│
├── frames/
│   ├── detection/
│   │   ├── train/{clip_uid}/*.jpg ← detection training frames
│   │   └── val/{clip_uid}/*.jpg
│   └── retrieval/
│       └── visual_crops/
│           └── {split}/{clip_uid}/{annotation_uid}_{qs_id}.jpg  ← query images
│
├── labels/
│   └── yolo/
│       ├── train/{clip_uid}/*.txt ← YOLO format: class cx cy w h (normalized)
│       └── val/{clip_uid}/*.txt
│
├── embeddings/                    ← created by notebook 04 VM job
│   ├── clip/
│   │   ├── index/{clip_uid}.npy           ← (n_frames, 512) float32
│   │   ├── index/{clip_uid}_frames.npy    ← (n_frames,) int32 video frame numbers
│   │   └── queries.npy                    ← (18111, 512) visual crop embeddings
│   ├── blip/
│   │   ├── index/{clip_uid}.npy           ← (n_frames, 768) float32
│   │   └── queries.npy                    ← (18111, 768)
│   └── queries_metadata.parquet           ← maps embedding row → annotation_uid, qs_id, clip_uid
│
└── models/                        ← created by notebook 03
    ├── yolov8s_finetuned/best.pt
    └── detr_finetuned/best.pt
```

---

## Key Data Concepts

### The Three Frame Types

**1. Detection frames** (`frames/detection/`)  
Frames extracted from the response track — these show where the object **was** before it disappeared. Used to train YOLOv8 and DETR. Each frame has a corresponding YOLO label file.

**2. Visual crops** (`frames/retrieval/visual_crops/`)  
A cropped image of the target object, extracted from a frame **after** the query frame. This is the query image for retrieval — "find me a frame that contains this object."

**3. Index frames** (embedded by VM script, not saved as JPEGs)  
Frames sampled at 1 FPS from each clip's source video. These form the FAISS search index — the retrieval model searches these to find where the object was last seen.

### Query-Set Structure
Each query-set represents one retrieval task:
- `query_frame`: clip-relative frame number where the object is **absent**
- `query_video_frame`: same, but indexed from source video start
- `visual_crop`: the query image (what we're searching for)
- `response_track`: ground truth — bounding boxes showing where the object was before disappearing

### Frame Number Conventions
- `frame_number`: clip-relative, at 5 FPS (annotations were drawn on 5 FPS clips)
- `video_frame_number`: source video-relative, at 30 FPS
- GCS videos are keyed by `video_uid`, not `clip_uid`
- Always use `video_frame_number` when reading from source videos

---

## GCP VM Setup

**VM name:** `egorecall-downloader`  
**Zone:** `us-central1-a`  
**Project:** `visual-memory-acv-project`

### Connect to VM
```bash
gcloud compute instances start egorecall-downloader --zone=us-central1-a
gcloud compute ssh egorecall-downloader --zone=us-central1-a
```

### Stop VM (important — costs money when running)
```bash
gcloud compute instances stop egorecall-downloader --zone=us-central1-a
```

### Run embedding job (for notebook 04 Option A)
```bash
# On the VM
gsutil cp gs://egorecall-data/processed/embed_index_frames.py ~/
gsutil cp gs://egorecall-data/ego4d/v2/annotations/vq_train.json /tmp/
gsutil cp gs://egorecall-data/ego4d/v2/annotations/vq_val.json /tmp/

pip3 install google-cloud-storage transformers torch torchvision opencv-python-headless tqdm

tmux new -s embed
python3 embed_index_frames.py
# Detach: Ctrl+B D
# Reattach: tmux attach -t embed
```

The script has a checkpoint — if interrupted, restart and it picks up where it left off.

---

## Running Notebook 04 — Two Options

### Option A: Full Dataset (pre-computed embeddings)
**Prerequisite:** VM embedding job must complete first (~4-6 hours with GPU).  
**Notebook:** `04_retrieval_baseline.ipynb`

Steps:
1. Start VM, run embedding script in tmux, let it run overnight
2. While VM runs, run §3 of the notebook in Colab to embed visual crops (~20-30 min)
3. After VM job completes, run §4-6 for evaluation

### Option B: Subset (self-contained, no VM needed)
**Prerequisite:** None — everything runs in Colab.  
**Notebook:** `04_retrieval_baseline_subset.ipynb`

Steps:
1. Open notebook in Colab with GPU runtime
2. Run all cells — extracts frames on-the-fly from 500 random val clips
3. Results available in ~1-2 hours

Option B is recommended for getting a first result quickly. Option A gives the full picture.

---

## Dependencies

```bash
pip install ultralytics transformers torch torchvision faiss-gpu \
    google-cloud-storage tqdm pandas pyarrow opencv-python-headless \
    pycocotools matplotlib seaborn pillow
```

---

## Colab Tips

- Use **GPU runtime** for notebooks 03, 04 (T4 is sufficient)
- Use **CPU runtime** for data syncing (saves GPU quota)
- Colab Pro gives ~200GB disk — enough for detection frames (~19GB)
- Detection frames were synced using tar files to avoid slow small-file transfers:
  ```
  gs://egorecall-data/processed/det_train.tar.gz   (~15GB)
  gs://egorecall-data/processed/det_val.tar.gz     (~5GB)
  gs://egorecall-data/processed/lbl_train.tar.gz
  gs://egorecall-data/processed/lbl_val.tar.gz
  ```

---

## Contact

Questions about infrastructure, data pipeline, or GCS access — reach out to Khadija.
