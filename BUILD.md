# Adaptive Pedestrian Detection Pipeline — Speed Without Sacrificing Safety

---

## Working Protocol (Non-Negotiable)

This project is built **incrementally**. Every piece of logic is tested and validated before moving to the next. No skipping ahead.

### The Rule
```
Write code → Test it → Validate metrics → Only then move forward
```

### What "Validated" Means at Each Stage
- **Fine-tuning**: Check validation loss curve, confirm recall improved over pretrained baseline on a small Caltech sample before running full eval
- **Baseline eval**: Confirm numbers make sense — YOLOv8l should have higher recall than YOLOv8n. If not, something is wrong, stop and debug
- **Scheduler**: Confirm % heavy triggers is reasonable (not 0%, not 100%) before sweeping thresholds
- **Plots**: Every plot must have clean axes, labels, title, and a one-line takeaway written below it
- **Demo video**: Must show light/heavy indicator switching visibly — if it never switches, threshold is wrong

### What Gets Updated After Every Step
- `results/` folder with actual measured numbers
- This README's results table with real values (no placeholders left behind)
- A commit with a human, descriptive message

### Commit Message Convention
Every commit follows this format:
```
[stage] what you did — what you found

Examples:
[baseline] ran YOLOv8n + YOLOv8l on 5 Caltech sequences — YOLOv8l misses 2x fewer occluded pedestrians
[finetune] fine-tuned both models on CrowdHuman — recall improved 8% on partial occlusion class
[scheduler] implemented confidence threshold T=0.35 — triggers heavy model on 22% of frames
[eval] swept thresholds 0.25-0.55 — T=0.45 gives best FPS/recall tradeoff
[demo] recorded demo video — light/heavy switching clearly visible at crowd scenes
```

### What "Done" Looks Like
- Clean metrics table with real numbers filled in
- 3 plots: FPS vs MD@100, occluded recall comparison, % heavy triggers vs MD@100
- 30 second demo video showing scheduler switching models in real time
- README tells a complete, quantitative story — no vague claims anywhere

---

## Build Log

### ✅ Day 1 — Environment Setup COMPLETE
**Date:** Day 1  
**Commit tag:** `[setup] environment confirmed — A100 ready, datasets in place`

**What was done:**
- Created full project directory structure
- Installed all dependencies via requirements.txt
- Downloaded and verified YOLOv8n (3.1M params) and YOLOv8l (43.7M params) via Ultralytics
- Confirmed CUDA available on NVIDIA A100-SXM4-80GB (80GB VRAM)
- Caltech Pedestrian dataset: all `.seq` files (set00–set10) + `.vbb` annotations confirmed present (Train + Test splits, 274 files total)
- CrowdHuman dataset: train01/02/03 + val images + `annotation_train.odgt` + `annotation_val.odgt` confirmed present

**Hardware confirmed:**
```
GPU: NVIDIA A100-SXM4-80GB
CUDA: True
YOLOv8n params: 3,157,200
YOLOv8l params: 43,691,520
```

**Status:** ✅ All systems go. Proceeding to Day 2.

---

### ✅ Day 2 — .seq Converter + Pretrained Baseline Eval COMPLETE
**Commit tag:** `[baseline] ran YOLOv8n + YOLOv8l pretrained on 5 Caltech sequences (set00) — domain gap confirmed, recall ~23-28% on real driving video`

**What was done:**
- Wrote `.seq` → JPEG frame extractor (JPEG SOI marker scanning approach)
- Extracted 500 frames × 5 sequences = 2,500 frames from set00 (V000–V004)
- Wrote `.vbb` annotation parser using scipy MATLAB loader
- Confirmed annotations: set00 has 16,677 total pedestrian instances across 5 sequences
- Ran pretrained YOLOv8n and YOLOv8l on all 2,500 frames with IoU ≥ 0.5 matching
- Both sanity checks passed

**Key finding — Domain Gap:**
Pretrained COCO models perform poorly on Caltech driving video. Recall is only 23–28%. This is expected — COCO underrepresents dense urban pedestrians from a moving vehicle perspective. Fine-tuning on CrowdHuman is the fix.

**Pretrained Baseline Numbers (set00, 2500 frames):**

| Model | FPS | Recall | MD@100 | Miss Rate (none occl) | Miss Rate (partial occl) |
|---|---|---|---|---|---|
| YOLOv8n pretrained | 124.5 | 0.2279 | 143.28 | 0.7232 | 0.9455 |
| YOLOv8l pretrained | 99.3 | 0.2774 | 134.08 | 0.6684 | 0.9126 |

**Observations:**
- YOLOv8l recall > YOLOv8n ✅ (sanity check passed)
- YOLOv8n FPS > YOLOv8l ✅ (sanity check passed)  
- Precision is high (76–77%) — when models detect, they're correct; the problem is entirely missed detections
- No heavily occluded instances in set00 — expected, set00 is a relatively clean sequence
- MD@100 of 143 means ~1.4 missed pedestrians per frame on average — dangerous for AV use

**Status:** ✅ Domain gap confirmed. Proceeding to Day 3–4: CrowdHuman fine-tuning.

---

### ⏳ Day 3–4 — CrowdHuman Fine-tuning
*In progress*

---

## The Problem

Modern object detection models face a brutal tradeoff in real-world deployment:

- Run a **heavy model** (YOLOv8l) → high accuracy, but too slow for real-time use (8–12 FPS)
- Run a **lightweight model** (YOLOv8n) → fast (30+ FPS), but misses occluded or edge-case pedestrians

In Autonomous Vehicle (AV) and UAV perception pipelines, this isn't just an accuracy problem — **it's a safety problem.** A model running at 10 FPS on a vehicle moving at 60 km/h means the perception system updates every 6 meters. That's the difference between stopping in time and not.

The question nobody cleanly answers: **At what inference speed does pedestrian detection become dangerous — and what can we do about it?**

---

## What This Project Does

This project does two things:

### 1. Quantifies the Safety Cost of Speed
Instead of reporting just FPS or generic accuracy metrics, this project introduces a safety-focused framing:
- **Missed Detections per 100 Frames (MD@100)** — how many pedestrians does the system silently miss as you push for speed?
  - *MD@100 = (total missed GT pedestrians / total frames) × 100, using IoU ≥ 0.5 matching.*
- Empirically identifies the FPS region where detection reliability degrades
- Separates failure modes: occluded vs small/far vs crowded scenes

### 2. Fixes It With a Simple, Deployable Pipeline
Once the breaking point is identified, a **confidence-triggered cascaded inference pipeline** is built:
- Runs YOLOv8n on every frame (fast, lightweight)
- If confidence is low, no pedestrian detected after a streak, or box is very small — triggers YOLOv8l on the full frame
- YOLOv8l result replaces YOLOv8n result for that frame
- Result: near-realtime FPS with heavy-model reliability on the frames that actually matter

---

## The Core Insight

> "The problem isn't that heavy models are slow or light models are inaccurate. The problem is that we run the same model uniformly on every frame regardless of what's happening in the scene. A smarter pipeline knows when to be fast and when to be careful."

---

## Datasets

### Training — CrowdHuman
- 15,000 training images, 4,370 validation images
- Specifically designed for crowded, occluded pedestrian scenarios
- Each instance annotated with full-body box (fbox), visible box (vbox), head box (hbox)
- Download: https://www.crowdhuman.org/

### Evaluation — Caltech Pedestrian
- Real video sequences captured from a moving vehicle in urban traffic
- `.seq` video files + `.vbb` annotation files per sequence
- Occlusion labels per instance: none, partial, heavy
- Recorded at 30 FPS
- Download: http://www.vision.caltech.edu/datasets/caltech_pedestrian/

---

## Models Used

| Model | Role | Why |
|---|---|---|
| YOLOv8n (fine-tuned) | Fast model, runs every frame | Lightweight, fine-tuned on CrowdHuman hard cases |
| YOLOv8l (fine-tuned) | Heavy model, triggered on uncertainty | High recall on occluded objects, fine-tuned on CrowdHuman |
| RT-DETR (optional) | Transformer-based comparison | Tests if attention mechanism handles occlusion better |

---

## Pipeline Architecture

```
Input Frame
     │
     ▼
YOLOv8n (full frame, every frame)
     │
     ├── High confidence + pedestrian detected + normal box size
     │   → Trust YOLOv8n result, skip heavy model
     │
     └── Any of these triggers:
         - confidence < threshold T
         - no pedestrian detected but previous frames had pedestrians
         - box is very small (far / hard pedestrian)
              │
              ▼
         YOLOv8l (full frame, only triggered frames)
              │
              ▼
         Final Detection Output
```

---

## Full Results Table

| Config | FPS | Recall | MD@100 | Miss Rate (partial occl) |
|---|---|---|---|---|
| YOLOv8n pretrained | 124.5 | 0.2279 | 143.28 | 0.9455 |
| YOLOv8l pretrained | 99.3 | 0.2774 | 134.08 | 0.9126 |
| YOLOv8n fine-tuned | — | — | — | — |
| YOLOv8l fine-tuned | — | — | — | — |
| **Adaptive Pipeline** | — | — | — | — |

*Evaluated on Caltech set00, 2500 frames, IoU ≥ 0.5. Fine-tuned rows will be filled after Day 5.*

---

## Project Structure

```
adaptive-pedestrian-detection/
│
├── data/
│   ├── caltech/              # .seq files + .vbb annotations (set00–set10)
│   └── crowdhuman/           # train01/02/03 + val images + .odgt annotations
│
├── models/
│   └── download_models.py
│
├── src/
│   ├── baseline.py
│   ├── scheduler.py
│   └── evaluate.py
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_adaptive.py
│   ├── sweep_thresholds.py
│   └── plot_results.py
│
├── results/
│   ├── baseline_metrics.csv
│   ├── adaptive_metrics.csv
│   └── figures/
│
├── demo/
│   └── demo_video.py
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python 3.10+
- Ultralytics YOLOv8
- OpenCV (video decoding + visualization overlays)
- NumPy, Pandas (metrics)
- Matplotlib / Seaborn (plots)
- PyTorch (model inference)

---

## References

- CrowdHuman: Shao et al., "CrowdHuman: A Benchmark for Detecting Human in a Crowd", arXiv 2018
- Caltech Pedestrian Dataset: Dollár et al., "Pedestrian Detection: An Evaluation of the State of the Art", PAMI 2012
- YOLOv8: Ultralytics, 2023
- RT-DETR: Wenyu Lv et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024