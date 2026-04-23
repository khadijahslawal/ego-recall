# EgoRecall: Egocentric Visual Memory Retrieval for Smart Glasses
 
---
 
## Project Overview
 
Smart glasses continuously index what the wearer sees. When the user needs to find something, they ask in natural language. The system searches the visual memory stream and returns the frame where the object was last seen.
 
**Example:** *"Where did I put my keys?"* → returns timestamped frame from 20 minutes ago showing keys on the kitchen counter.
 
### Two Cognitive Problems
 
| | Problem | Champion | Challenger | Metrics |
|---|---|---|---|---|
| P1 | Object detection | YOLOv8 | DETR | mAP@50, mAP@75, precision/recall |
| P2 | Visual retrieval | CLIP + FAISS | BLIP + FAISS | Recall@1, Recall@5, MRR |
 
### Dataset
 
**Ego4D** — Meta/CMU egocentric video benchmark  
3,600+ hours of first-person video across 74 worldwide locations  
License required: https://ego4d-data.org/docs/start-here/
 