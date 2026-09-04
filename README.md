# EgoRecall - Visual Memory Retrieval for Smart Glasses

<sub>**Authors:** Anais Morales &nbsp;·&nbsp; Estella Hu &nbsp;·&nbsp; Gaoyuan Gu &nbsp;·&nbsp; Khhadija</sub>

EgoRecall is an AI-powered visual memory system designed for smart glasses and wearable computing platforms. The system continuously indexes a user’s egocentric visual stream and enables retrieval of previously observed scenes through semantic search.

The project explores the question:

> “Can computer vision systems augment human episodic memory?”

Users can issue queries such as:
- “Where did I leave my keys?”
- “Show me the frame where I routed the wiring harness.”
- “Did I lock the door?”

The system retrieves the most relevant historical frame along with timestamps and object localization results.

Built using the Ego4D Visual Object Queries (VQ) benchmark, EgoRecall combines:
- egocentric object detection,
- multimodal visual retrieval,
- vector similarity search,
- and scalable embedding indexing

into a unified wearable AI memory pipeline.

## Key Technical Contributions

- Designed a two-stage multimodal visual memory pipeline for wearable AI systems
- Built a scalable FAISS-based vector retrieval architecture
- Compared CNN-based and transformer-based object detectors under constrained compute settings
- Evaluated multimodal embedding approaches for egocentric episodic memory retrieval
- Developed preprocessing and indexing pipelines for large-scale Ego4D video data
- Implemented deployment-oriented local-first retrieval architecture for privacy preservation


---
- [EgoRecall - Visual Memory Retrieval for Smart Glasses](#egorecall---visual-memory-retrieval-for-smart-glasses)
  - [Key Technical Contributions](#key-technical-contributions)
  - [Business Problem](#business-problem)
  - [Dataset](#dataset)
    - [Dataset Statistics](#dataset-statistics)
  - [Data Infrastructure - The Real Bottleneck](#data-infrastructure---the-real-bottleneck)
    - [1. Dataset Access and Download to GCS](#1-dataset-access-and-download-to-gcs)
    - [2. Discovering the Actual GCS Path Structure](#2-discovering-the-actual-gcs-path-structure)
    - [3. Frame Extraction Architecture and Optimization](#3-frame-extraction-architecture-and-optimization)
    - [4. The Annotation Resolution Mismatch](#4-the-annotation-resolution-mismatch)
    - [5. The Small-File GCS Transfer Problem](#5-the-small-file-gcs-transfer-problem)
    - [6. Colab Session Timeouts and Training Continuity](#6-colab-session-timeouts-and-training-continuity)
    - [7. VM SSH Instability](#7-vm-ssh-instability)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [System Architecture](#system-architecture)
    - [Stage 1: Object Detection](#stage-1-object-detection)
    - [Stage 2: Visual Retrieval](#stage-2-visual-retrieval)
  - [Cognitive Problem A: Object Detection](#cognitive-problem-a-object-detection)
    - [Goal](#goal)
    - [Champion Model - YOLOv8s](#champion-model---yolov8s)
    - [Challenger Model - Deformable DETR](#challenger-model---deformable-detr)
    - [Key Challenges in Object Detection Overall](#key-challenges-in-object-detection-overall)
  - [Cognitive Problem B - Visual Retrieval](#cognitive-problem-b---visual-retrieval)
    - [Goal](#goal-1)
    - [Champion Model](#champion-model)
    - [Challenger Model](#challenger-model)
    - [Retrieval Challenges](#retrieval-challenges)
    - [Visual Retrieval Pipeline Deep Dive](#visual-retrieval-pipeline-deep-dive)
      - [Important: Two Different Evaluation Metrics](#important-two-different-evaluation-metrics)
      - [Compute Infrastructure - Why a GCP Vertex AI Workbench](#compute-infrastructure---why-a-gcp-vertex-ai-workbench)
      - [Cache Architecture](#cache-architecture)
      - [Ground-Truth Coverage Upper Bound](#ground-truth-coverage-upper-bound)
      - [Retrieval Experiments - Progressive Pipeline](#retrieval-experiments---progressive-pipeline)
      - [Why Results Are Low - Honest Interpretation](#why-results-are-low---honest-interpretation)
  - [Deployment Architecture](#deployment-architecture)
  - [Future Improvements](#future-improvements)

## Business Problem

Modern users process enormous amounts of visual information, yet only a fraction reaches conscious memory. While tools such as Microsoft Recall and Rewind.ai index digital screen activity, no mainstream system currently indexes what users physically observe in the real world.

EgoRecall explores how wearable AI systems could bridge this gap by transforming continuous first-person video into a searchable visual memory system.

Potential applications include:
- aerospace assembly validation,
- surgical workflow review,
- industrial inspection,
- and accessibility support.

---
## Dataset


The dataset used in this project is the [Ego4D Visual Object Queries (VQ)](https://ego4d-data.org/) - a large-scale egocentric video dataset with over 3,000 hours of videos covering hundreds of scenarios (household, outdoor, workplace, leisure e.t.c) of daily life activity captured in the wild.

The benchmark specifically used in this project is part of the Episodic Memory Benchmark which aims to make past video queryable and requires localizing where the answer can be seen within the user’s past video. 

### Dataset Statistics

The table below provides a high-level of the scale of the data that was utilized for this project. A huge portion of the work ended up being on the data processing from storage to transfer and extraction. 

|  | Value |
| :--- | :--- |
| Unique Videos | 1,743 |
| Data Size | 667 GB|
| Valid Query Sets| 18,114|
| Unique Objects | 3,487|
| Extracted Frames | 246,206|
| Visual Crop Images| 18,111|

In the subsequent section; detailed information is provided on the data-processing 

## Data Infrastructure - The Real Bottleneck
 
This section documents the infrastructure work in detail as it consumed the majority of project time and involved non-trivial engineering decisions at every step. It is the foundation everything else runs on.

### 1. Dataset Access and Download to GCS
 
The Ego4D dataset requires an access request through Meta AI Research. Once approved, videos were downloaded via the Ego4D CLI filtered to the Visual Object Queries (VQ) benchmark subset. The full VQ benchmark spans 667GB across 1,743 unique source videos.
 
The download was run on a GCP VM using `tmux` to maintain persistent sessions independent of SSH connectivity and downloaded directly to a dedicated GCS bucket  (`gs://egorecall-data/`) rather than VM local disk, since the VM's local disk wouldn't hold 667GB. The GCS bucket served as the single source of truth for all raw data throughout the project.

### 2. Discovering the Actual GCS Path Structure
 
The first non-trivial infrastructure problem was that **the videos were not where the documentation suggested.**
 
Initial attempts to access videos by `clip_uid` failed with 404 errors. Debugging revealed three layers of mismatch:
 
**Problem 1: Videos are nested under `ego4d/v2/`:**
```
# Expected (wrong)
gs://egorecall-data/{clip_uid}.mp4
 
# Actual
gs://egorecall-data/ego4d/v2/video_540ss/{video_uid}.mp4
```

**Problem 2: Videos are keyed by `video_uid`, not `clip_uid`**

The Ego4D annotation JSON contains both `clip_uid` and `source_clip_uid` fields, neither of which matched the GCS filenames. The actual GCS files use `video_uid` as the top-level video identifier, one level above clips in the annotation hierarchy.
 
This required cross-referencing the annotation structure against actual GCS blob names:
```python
# List actual GCS filenames
blobs = list(gcs_client.list_blobs(BUCKET_NAME, prefix="ego4d/v2/video_540ss/"))
gcs_uids = set(b.name.split("/")[-1].replace(".mp4", "") for b in blobs)
 
# Cross-reference against annotation fields
video_uid_matches = set(df["video_uid"]) & gcs_uids   # → 1,727 matches
clip_uid_matches  = set(df["clip_uid"])  & gcs_uids   # → 0 matches
```

**Problem 3: Multiple clips per source video:**

The annotation structure maps multiple `clip_uid` entries to the same `video_uid`. 

Distribution:
```
clips per video: 1→633, 2→482, 3→216, 4→132, 5→78 ... up to 18
average: 2.7 clips per video
```
This meant a naive per-clip download loop would download the same source video 2-3 times on average which could have led to ~2,963 redundant downloads. This was avoided by restructuring the workplan to group by `video_uid` and process all clips in a single video in one pass.

### 3. Frame Extraction Architecture and Optimization
 
The extraction script (`02_frame_extraction.ipynb`, later converted to `train_yolo.py` for VM execution) went through several iterations before reaching acceptable performance.
 
**Initial approach - random seek per frame:**
```python
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
```
- Speed: ~2.7 frames/second at 266,771 frames needed
- Estimated runtime: ~27 hours. 

This approach was not viable.
 
**Iteration 2 - Sequential scan:**

The second approach considered reading every frame and keeping only the needed ones. It is faster for dense frame sets but still slow when frames are sparsely distributed across long videos (e.g 0 to frame 22,440 with 30-frame gaps between needed frames).
 
**Iteration 3 - Cluster-seek hybrid:**

Here, we grouped needed frame numbers into clusters of nearby frames (gap threshold: 150 frames = 5 seconds). Then applied a sequential read within each cluster and a seek between clusters. This reduced the extraction time from 205s per video to 74s per video.
 
**Final approach - Separate detection/VC frames from index frames:**

The index frames (1 FPS sampling across entire video) dominated runtime because they required scanning 22,440 frames to collect ~557 frames. Separating the two extraction types and deferring index frames to retrieval time reduced per-video extraction to ~3s for detection + visual crop frames only.
 
**Final timing benchmark (Colab T4, CPU extraction):**
```
Download     : ~4s   per video
Extraction   : ~3s   (detection + visual crop frames only)
Upload       : ~5s
Total        : ~12s  × 1,727 videos ≈ 5.7 hours
```
### 4. The Annotation Resolution Mismatch
 
This was caught during the EDA dry run, not during training, which is the only reason it didn't corrupt the 246K training labels.
 
The Ego4D annotations were drawn on 1440×1080 frames. The actual video files (`video_540ss`) were stored at 720×540 which was exactly 2× downscaled. Every bounding box coordinate in the annotation JSON is in 1440×1080 space.
 
Discovery: when the first visual crop was extracted, it returned an empty array. Inspecting the frame dimensions revealed the mismatch:
 
```python
print(f"Frame shape (actual)    : {frame.shape}")    # (540, 720, 3)
print(f"Annotation orig_w×orig_h: {vc['original_width']}×{vc['original_height']}")  # 1440×1080
print(f"Crop y coordinate       : {vc['y']}")         # 587 - out of bounds for 540px frame
```
 
Fix applied to both `crop_image()` and `make_yolo_label()`:
```python
scale_x = actual_w / orig_w  # = 720 / 1440 = 0.5
scale_y = actual_h / orig_h  # = 540 / 1080 = 0.5
```
 
Without this fix, every visual crop would be empty and every YOLO label would place bounding boxes outside the frame bounds.

### 5. The Small-File GCS Transfer Problem
 
Lastly, with 246,206 detection frames extracted and labeled, the next task was syncing them to Colab for training. `gsutil -m rsync` ran for 2+ hours and still hadn't finished train images alone. The overhead of transferring 246K individual small files (averaging 80KB each) dominated over the actual data transfer rate.
 
**Root cause:** GCS charges per-operation as well as per-byte. Syncing 246K files means 246K individual GET/PUT operations regardless of file size. At typical GCS throughput, the metadata overhead per small file far exceeds the transfer time.
 
**Solution - tar bundling on the GCP VM:**
Rather than transferring files individually, the VM (which has fast intra-GCP network to GCS) was used as an intermediary:
 
```bash
# On VM: copy from GCS, tar, upload tar back to GCS
gsutil -m cp -r gs://egorecall-data/frames/detection/train/* ~/det_train/
tar -cf ~/det_train.tar ~/det_train/   # no compression - JPEGs don't compress
gsutil cp ~/det_train.tar gs://egorecall-data/processed/det_train.tar
 
# In Colab: download 1 file instead of 184K
!gsutil cp gs://egorecall-data/processed/det_train.tar /tmp/
!tar -xf /tmp/det_train.tar -C /content/egorecall_detection/train/images/ --strip-components=3
```
 
**Why `--strip-components=3`:** The tar was created from `~/det_train/` which expands to `/home/khadijahlawal/det_train/{clip_uid}/{frame}.jpg`. Without stripping, extraction would recreate the full home directory path. `--strip-components=3` removes `home/khadijahlawal/det_train/` leaving clean `{clip_uid}/{frame}.jpg`.
 
**Why no compression (`-cf` not `-czf`):** JPEGs are already compressed. Adding gzip would take 10-15 minutes to reduce file size by ~1-2%. It was not worth it.
 
Transfer time: 4 files downloaded in minutes vs 2-3 hours for 246K individual files.

### 6. Colab Session Timeouts and Training Continuity
 
YOLOv8 training at `imgsz=640` estimated ~1:22 per epoch on T4. Thus, 50 epochs would take ~69 hours, far beyond any Colab session limit. Even at `imgsz=320` with 20 epochs, the ~7-hour estimate exceeded typical free session limits.
 
**Solutions applied:**
 
**Browser keep-alive script** (must run before starting training):
```javascript
%%javascript
function KeepAlive() {
    document.querySelector("#top-toolbar > colab-connect-button")
            .shadowRoot.querySelector("#connect").click();
}
setInterval(KeepAlive, 60000);
```
 
**Epoch-level checkpointing with `save_period=1`:**
```python
yolo_finetune.train(save_period=1, ...)  # saves last.pt every epoch
```
 
**Resume from checkpoint:**
```python
# Detects last.pt and resumes automatically
if last_pt.exists():
    model = YOLO(str(last_pt))
    model.train(resume=True)
```
 
**Google Drive saving every epoch** (for DETR, which doesn't have built-in resume):
```python
torch.save(detr_model.state_dict(), "/content/drive/MyDrive/detr_last.pt")
```
Drive persists across Colab sessions even when `/content/` is wiped.

### 7. VM SSH Instability
 
The GCP VM experienced recurring SSH connection failure that required a stop/start cycle to resolve. This happened multiple times during overnight jobs, interrupting the embedding script mid-run.
 
Root cause was never definitively identified (likely GCP's automatic maintenance migration combined with the VM's e2-standard-2 machine type). The checkpoint system in all VM scripts (`embed_index_frames.py`, `train_yolo.py`) was designed specifically to survive these interruptions. Every 50 videos completed writes a JSON checkpoint so the script restarts cleanly from where it left off.
 
**Important VM persistence note:** `/tmp` is cleared on VM restart. Any files written there (annotation JSONs, intermediate data) must be re-downloaded after a restart. All persistent data was written to `~/data/` or directly to GCS.
 
---
## Exploratory Data Analysis


Full Details on EDA is in `01_dataset_overview.ipynb` 
| What the data showed| What it pushed us towards |
| :--- | :--- |
| 1.9% median object area - COCO detectors fail out-of-the-box (diagnostics) | A detector that makes no prior assumptions about object shape or scale|
| 3,487 open-vocabulary object types, most appear fewer than 5 times |Treating detection as "is there an object here?" rather than "which of N classes is this?" |
| Temporal gap up to 4,750 frames (~15 min)| Searching the entire clip history, not just a recent window (FAISS) |
| Annotations at 1440×1080, frames at 720×540| Coordinate correction step before any labeling |

[Insert: Image of Median Object Area]

[Insert: Image of top 10 objects]

---

## System Architecture

The pipeline operates in two stages:

### Stage 1: Object Detection
Egocentric video frames are processed using object detection models to localize visible objects and generate structured visual memory logs.

### Stage 2: Visual Retrieval
Frames are embedded using multimodal vision-language models and stored in a FAISS vector index. User queries are embedded into the same representation space and matched through nearest-neighbor similarity search.

Pipeline:

Egocentric Video Stream  → Object Detection   → Embedding Generation   → FAISS Vector Index  → Semantic Retrieval   → Retrieved Memory Frame + Timestamp

---

## Cognitive Problem A: Object Detection

### Goal
Given an egocentric video frame, we want to detect and localize the target object with a bounding box

### Champion Model - YOLOv8s

The YOLOv8s was selecyed for  optimized for real-time inference, anchor-free detection, edge-device feasibility (in the case of real-world deployment) and a strong accuracy-speed tradeoff.

### Challenger Model - Deformable DETR

Deformable DETR is transformer-based detection model with  multi-scale deformable attention and stronger contextual reasoning for partially occluded objects.

**Augmentation Applied for Object Detection Pipeline**

| Augmentation | Value | Rationale |
| :--- | :--- | :--- |
| **Mosaic** | 1.0 (always on) | Synthesizes multi-object scenes; addresses sparse detections per frame |
| **HSV jitter** | H=0.015, S=0.7, V=0.4 | Handles lighting variation across indoor/outdoor egocentric settings |
| **Horizontal flip** | 0.5 | Standard spatial invariance |
| **Vertical flip** | 0.0 (disabled) | Egocentric video has orientation meaning. Objects don't appear upside down |

**Evaluation Strategy for both models**

1. Baseline: Zero-shot (pretrained COCO weights, class-agnostic evaluation)
2. Fine-tuned Model 
3. Comparison between Zero-shot vs. fine-tuned for each model

**Metrics**
1. mAP@0.5: This is the primary metric and measures whether the model found the object at all.

2. mAP@0.5:0.95 : Averaged across IoU thresholds 0.5 - 0.95 in 0.05 steps. 
It penalizes loose boxes and is more meaningful for EgoRecall specifically as a poorly localized box produces a bad visual crop, which breaks retrieval downstream.


### Key Challenges in Object Detection Overall

[Insert Mr.Krisps Image]

- Objects occupy a median of 1.9% of frame area  far smaller than standard COCO benchmarks.
- Open vocabulary: 3,487 unique object types, no fixed class list, therefore he model must generalize, not memorize (e.g being able to differentiate between Mr.Krisps and a Pringles rather than simply memorizing "Chips" )

- Egocentric backgrounds are cluttered and dynamic as  the camera moves with the wearer; therefore this brings in a lot of extra detailed information the models have to consider in addition to having to objects the small objects 

- Domain gap: All pretrained detectors are COCO-trained which contain clean, upright, well-framed objects. None of which describes Ego4D



---

## Cognitive Problem B - Visual Retrieval

### Goal
The retrieval task: given a visual crop (a small cropped image of the target object), find the video frame where that object was last seen. This is a nearest-neighbor search problem in embedding space, but with several non-trivial challenges:

- The query is a small, often blurry cropped object (~39×60px at actual resolution)
- The index frames are full 720×540 egocentric frames with cluttered backgrounds
- 3,487 open-vocabulary object types; no class list to rely on
- Temporal gaps up to 4,750 frames (~15 minutes) between query frame and response track
- Only ~67% of queries have a ground-truth frame covered by the 1 FPS index sampling


### Champion Model
CLIP + FAISS
- efficient zero-shot multimodal retrieval,
- scalable embedding search,
- strong language-image alignment.

### Challenger Model
BLIP-ITM + FAISS
- richer semantic understanding,
- stronger fine-grained scene reasoning,
- improved image-text matching.

### Retrieval Challenges
- visually similar distractor objects,
- long temporal gaps,
- cluttered full-scene retrieval,
- egocentric motion blur,
- semantic ambiguity.

### Visual Retrieval Pipeline Deep Dive

This section documents the retrieval pipeline in full detail, including the preprocessing architecture, embedding strategy, and the progressive retrieval experiments that produced the final results.

#### Important: Two Different Evaluation Metrics

The retrieval results appear in two places with **very different numbers**, this is not an error. They use different tolerance thresholds that measure fundamentally different things:

| Source | Tolerance | Meaning |
|--------|-----------|---------|
| Presentation slides / notebook 04 | ±5 **seconds** (`TOLERANCE_FRAMES = 150`) | Temporal proximity - did we retrieve roughly the right moment? |
| Tuning notebook 07 | ±5 **frames** (`TOLERANCE_FRAMES = 5`) | Precise frame localization - did we retrieve the exact frame? |

At 30 FPS source video, ±5 seconds = ±150 video frames. At ±5 frames, you need to land within 0.17 seconds of the response track. The tuning notebook is ~30× stricter.

**BLIP Top-1 accuracy:**
- Slides metric (±5 seconds): **8.64%**
- Tuning metric (±5 frames): **0.52%**

Both numbers are correct for what they measure. The slides metric is closer to the real user experience - "show me roughly when I last had this object" and is the more meaningful product metric. The tuning metric tests precise localization, which is a much harder problem and is more relevant if detection bounding boxes are needed for the final output.

When reading the tuning experiment results below, keep in mind they use the strict ±5 frame threshold.

#### Compute Infrastructure - Why a GCP Vertex AI Workbench

The retrieval preprocessing (`06_preprocess_nclips300`) ran on a **GCP Vertex AI Workbench instance** (`/home/jupyter/` paths in the notebooks), not on Colab. This was a deliberate choice driven by the scale of the job:

- 300 clips × ~300 index frames each = ~90,000 frames to embed through CLIP and BLIP
- Each clip required downloading a ~400MB source video, extracting frames, embedding, saving cache files, then deleting the video
- The Workbench instance provided persistent storage and longer-running sessions than Colab for this multi-hour preprocessing job

The preprocessing was split into 5 batches of 60 clips each (`BATCH_ID = 0, 1, 2, 3, 4`) so that each batch could be run independently and the cache files saved to disk incrementally. This was important because the job was long enough that a single session failure would otherwise lose all progress.

#### Cache Architecture

The preprocessing notebook produces a structured cache for each clip:

```
results/retrieval_subset/cache/
├── index_frame_embeddings/
│   └── {clip_uid}_index_embeddings.npz    ← frame_numbers, clip_embs, blip_embs
└── query_embeddings/
    ├── {clip_uid}_query_embeddings.npz    ← clip_embs, blip_embs
    └── {clip_uid}_query_metadata.parquet  ← annotation_uid, qs_id, object_title
```

Each `.npz` file stores both CLIP (512-dim) and BLIP (768-dim) embeddings together, which allowed the tuning notebook to load both models' embeddings in a single file read rather than two separate reads per clip. This was a design decision that paid off during the grid search experiments, where each clip was loaded dozens of times.

**Robustness - targeted repair rather than full re-run:**
After the 5 batches completed, one clip (`18dad6e7-5969-4573-a1b3-f4ccfc53c350`) failed to cache correctly. Rather than re-running the entire preprocessing job, a `preprocess_one_clip_for_cache()` repair function was used to reprocess just that single clip. The final cache completeness check (`check_clip_cache_status()`) confirmed all 300 clips were fully cached before proceeding to tuning. This pattern starting from verifying completeness, repairing selectively and re-verifying is worth keeping for any future preprocessing job at this scale.

**Why cache embeddings rather than recompute on the fly:**  
CLIP and BLIP inference on ~300 frames per clip takes ~10-30 seconds per clip on GPU. With 5 retrieval methods × multiple hyperparameter settings to evaluate, recomputing would have taken hours. Caching the embeddings once reduced each evaluation run to pure numpy operations such as loading, dot products, sorting which runs across 300 clips in under a minute.

**Defensive embedding extraction:**
The `_to_feature_tensor()` helper handles inconsistent output formats across CLIP and BLIP HuggingFace versions. It tries `image_embeds`, then `pooler_output`, then `last_hidden_state[:, 0, :]` before raising an error. This prevents silent failures when model output formats change across library updates, which happened during development.


#### Ground-Truth Coverage Upper Bound

Before running any retrieval experiments, the team computed a critical diagnostic: what fraction of queries even have a ground-truth frame in the index?

The 1 FPS sampling strategy means index frames are spaced 30 video frames apart. If a response track falls entirely between two sampled frames, no retrieval method can succeed regardless of embedding quality. The analysis showed:

**~67% of queries have at least one index frame within ±5 seconds of the response track.**

This 67% figure is the theoretical upper bound on Top-100 retrieval accuracy. The ~33% gap is not a model failure, it's a sampling coverage failure. The practical implication: even a perfect retrieval model can only achieve ~67% Top-100 accuracy with 1 FPS indexing. Increasing to 2 FPS would substantially close this gap.

This finding shaped the interpretation of all subsequent results. A Top-100 accuracy of 25% should be compared against the 67% upper bound, not against 100%.

#### Retrieval Experiments - Progressive Pipeline

The tuning notebook (`07_tuning_nclips300_cached_retrieval`) evaluated five increasingly sophisticated retrieval strategies, each building on the previous. All experiments operated on the same 300-clip, 1,157-query evaluation set.

**Evaluation metric:** `first_correct_rank` - the rank at which the first frame within ±5 frames of the response track appears. Success at Top-K means `first_correct_rank ≤ K`.

**Stage 1 - Single Model Baseline (CLIP vs BLIP)**

Raw cosine similarity between query embedding and index frame embeddings, using L2-normalized vectors so dot product = cosine similarity.

| Model | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 | Top-100 |
|-------|-------|-------|--------|--------|--------|---------|
| CLIP  | 0.52% | 1.73% | 2.94%  | 5.27%  | 15.04% | 24.98%  |
| BLIP  | 0.52% | 1.47% | 2.85%  | 5.62%  | 13.57% | 25.67%  |

CLIP and BLIP perform nearly identically. The complementary pattern showed that CLIP is better at Top-50, BLIP slightly better at Top-100. This suggested the two embedding spaces contain partially different signals, motivating fusion experiments.

**Stage 2 - CLIP/BLIP Alpha Fusion**

Linear interpolation of CLIP and BLIP similarity scores:

```
fusion_score = α × CLIP_score + (1 - α) × BLIP_score
```

Grid search over α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}.

| Alpha | Interpretation | Top-1 | Top-10 | Top-50 | Top-100 |
|-------|---------------|-------|--------|--------|---------|
| 0.00  | BLIP-only     | 0.52% | 2.85%  | 13.57% | 25.67%  |
| 0.50  | Equal fusion  | 0.61% | 2.77%  | 13.83% | 25.84%  |
| 0.75  | CLIP-dominant | 0.43% | 3.03%  | 14.17% | 25.50%  |
| 1.00  | CLIP-only     | 0.52% | 2.94%  | 15.04% | 24.98%  |

**Finding:** Fusion provides only marginal improvement over single-model baselines. The best Top-100 (25.84%) comes from α=0.5, but the gain over BLIP-only (25.67%) is minimal. The conclusion was that embedding-level fusion alone cannot close the recall gap. The bottleneck is whether the correct frame exists in the candidate set at all, not how it's ranked once it's there.

**Stage 3 - Temporal Smoothing**

After fusion, smooth similarity scores across neighboring sampled index frames:

```python
for i in range(n):
    window = scores[max(0,i-half) : min(n,i+half+1)]
    smoothed[i] = window.max()  # or window.mean()
```

The intuition: objects don't appear and disappear between consecutive 1-second sampled frames. If frame t has a high similarity score, neighboring frames t-1 and t+1 should also receive a score boost.

Grid search over: α ∈ {0.5, 0.75}, window_size ∈ {3, 5, 7}, mode ∈ {mean, max}.

**Finding:** Max smoothing consistently outperforms mean smoothing. Mean smoothing dilutes sharp local similarity peaks. This is  important because response tracks are short (median 10 frames) and the correct frame window can be narrow. Max smoothing preserves these local peaks.

Best temporal smoothing result: Top-20 improved to **7.00%** (vs 5.45% from alpha fusion), but Top-100 did not improve substantially. Temporal smoothing helps promote correct frames when they're nearby strong candidates, but doesn't solve the broader recall problem.

**Stage 4 - Candidate-Guided Window Reranking**

A more targeted approach to addressing the recall-precision tradeoff:

1. Compute fusion scores
2. Select top `candidate_topk` frames as anchors
3. Expand ±`window_radius` index positions around each anchor
4. Re-score the expanded candidate set

```python
anchor_indices = np.argsort(-fusion_scores)[:candidate_topk]
for anchor_idx in anchor_indices:
    for idx in range(anchor_idx - window_radius, anchor_idx + window_radius + 1):
        expanded_scores[idx].append(fusion_scores[idx])
```

The intuition: the top-k frames from raw similarity are likely near the correct answer temporally. Expanding around them increases the chance of including the actual ground-truth frame, which may have been missed by the 1 FPS sampling.

Grid search over: candidate_topk ∈ {10, 20, 50}, window_radius ∈ {2, 5, 10}.

Best result: Top-100 improved to **26.62%** (candidate_window outperforming pure temporal smoothing at the broad recall level).

**Stage 5 - Hybrid: Candidate Window + Local Temporal Smoothing**

Combining Stage 4 (candidate expansion) with Stage 3 (temporal smoothing within the expanded window):

```
Best configuration:
  alpha = 0.5
  candidate_topk = 50
  window_radius = 5
  local_smooth_radius = 5
  smooth_mode = max
```

**Final results (best method):**
- Top-20: **7.69%**
- Top-50: **16.85%**
- Top-100: **27.92%**

The hybrid method provides the best balance: candidate-window expansion improves broad candidate recall, while local temporal smoothing improves ranking within the expanded candidate regions.

#### Why Results Are Low - Honest Interpretation

The tuning notebook results (Top-1: 0.52%, Top-100: 27.92%) need to be read against three constraints:

**1. The ±5 frame threshold is extremely strict.**
The tuning experiments require retrieving within 0.17 seconds of the response track. Against the slides' ±5 second threshold, BLIP baseline achieves 8.64% Top-1 - a 16× difference from the same model. The tuning metric is appropriate for research benchmarking; the slides metric is appropriate for product evaluation. Neither is wrong, they measure different things.

**2. The 67% coverage ceiling.**
~33% of queries cannot succeed at any Top-K because no 1 FPS index frame falls within ±5 frames of their response track. This is a sampling artifact, not a model failure. Against the stricter metric, even a perfect retrieval model cannot exceed ~67% Top-100. Against the ±5 second metric this ceiling is higher, which is why slides numbers look more reasonable.

**3. Query-index domain mismatch.**
The query is a small cropped object image (~39×60px); the index frames are full 720×540 egocentric scenes. CLIP and BLIP were pretrained on web image-caption pairs, neither was trained to match a small crop against a full frame containing that crop somewhere.

**What would actually help:**
- 2-5 FPS index sampling to close the coverage gap
- Fine-tuning CLIP/BLIP contrastively on matched visual crop–full frame pairs from Ego4D
- Using the `object_title` text label as an additional query signal via CLIP's text encoder
- Using the ±5 second metric as the primary evaluation criterion for product-oriented work


---

## Deployment Architecture

Conceptual deployment pipeline:

Smart Glasses  
→ Frame Sampling  
→ Detection + Embedding  
→ Local FAISS Index Update  
→ User Query  
→ Semantic Retrieval  
→ Returned Frame + Timestamp

The system was designed with:
- local-first inference,
- privacy-preserving indexing,
- rolling-window retention policies,
- and embedding version management.

---

## Future Improvements

- Full-resolution YOLOv8 retraining on the complete 246K-frame dataset
- Contrastive fine-tuning of CLIP on Ego4D retrieval pairs
- Natural-language retrieval support
- GPU-based DETR training for fairer comparison
- Temporal retrieval reasoning across video sequences
- On-device optimization for wearable hardware
