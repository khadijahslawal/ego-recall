# EgoRecall - Visual Memory Retrieval for Smart Glasses

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

---

## Business Problem

Modern users process enormous amounts of visual information, yet only a fraction reaches conscious memory. While tools such as Microsoft Recall and Rewind.ai index digital screen activity, no mainstream system currently indexes what users physically observe in the real world.

EgoRecall explores how wearable AI systems could bridge this gap by transforming continuous first-person video into a searchable visual memory system.

Potential applications include:
- consumer memory assistance,
- aerospace assembly validation,
- surgical workflow review,
- industrial inspection,
- and accessibility support.

---

## System Architecture

The pipeline operates in two stages:

### Stage 1 — Object Detection
Egocentric video frames are processed using object detection models to localize visible objects and generate structured visual memory logs.

### Stage 2 — Visual Retrieval
Frames are embedded using multimodal vision-language models and stored in a FAISS vector index. User queries are embedded into the same representation space and matched through nearest-neighbor similarity search.

Pipeline:

Egocentric Video Stream  
→ Object Detection  
→ Embedding Generation  
→ FAISS Vector Index  
→ Semantic Retrieval  
→ Retrieved Memory Frame + Timestamp

---

## Cognitive Problem A — Object Detection

### Goal
Detect and localize objects appearing within egocentric video frames.

### Champion Model
YOLOv8s
- optimized for real-time inference,
- anchor-free detection,
- edge-device feasibility,
- strong accuracy-speed tradeoff.

### Challenger Model
Deformable DETR
- transformer-based detection,
- multi-scale deformable attention,
- stronger contextual reasoning for partially occluded objects.

### Key Challenges
- median object size only 1.9% of frame area,
- highly cluttered first-person environments,
- open-vocabulary object space (3,487 object categories),
- severe domain gap from standard COCO datasets.

---

## Cognitive Problem B — Visual Retrieval

### Goal
Given a query image or semantic concept, retrieve the frame where the object was last observed.

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

---

## Dataset & EDA Insights

Dataset: Ego4D Visual Object Queries (VQ)

Key dataset characteristics:
- 1,743 egocentric videos,
- 246K extracted frames,
- 18K+ valid query sets,
- 3,487 open-vocabulary object categories,
- 667GB dataset size.

EDA findings strongly influenced model design:
- objects occupied only 1.9% median frame area,
- temporal gaps reached ~15 minutes,
- annotation/frame resolution mismatches required coordinate correction,
- long-tail object distribution limited closed-set classification approaches.

---

## Key Technical Contributions

- Designed a two-stage multimodal visual memory pipeline for wearable AI systems
- Built a scalable FAISS-based vector retrieval architecture
- Compared CNN-based and transformer-based object detectors under constrained compute settings
- Evaluated multimodal embedding approaches for egocentric episodic memory retrieval
- Developed preprocessing and indexing pipelines for large-scale Ego4D video data
- Implemented deployment-oriented local-first retrieval architecture for privacy preservation

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