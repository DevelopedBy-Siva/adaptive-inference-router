# Adaptive Inference Router: Confidence-Based Model Selection for Real-Time Object Detection

Real-time perception systems usually face a simple tradeoff: run a **large model for accuracy** or a **small model for speed**. Most deployments pick one model and accept the compromise.

This project explores a different approach:

> **Can a system dynamically decide, frame by frame, when higher accuracy is worth the extra compute?**

The result is a **confidence-based inference router** that runs a lightweight model on every frame and selectively escalates uncertain frames to a heavier model. The system preserves near-lightweight throughput while improving detection reliability.

---

# Overview

The pipeline runs **YOLOv8n (light model)** on every frame. When the model appears uncertain — based on confidence, detection continuity, or object scale — the frame is escalated to **YOLOv8l (heavy model)**.

Only those frames are recomputed with the larger model, and the heavier model’s output replaces the lightweight result for that frame.

```
Frame → YOLOv8n
        |
        | confident detection
        └── use YOLOv8n result

        | uncertainty detected
        └── run YOLOv8l → replace detection
```

This creates a **compute-aware inference policy** rather than a fixed model choice.

The adaptive pipeline achieves:

* **124.6 FPS inference throughput**
* **0.8655 precision (highest among all tested configurations)**

while invoking the heavy model on only **~27% of frames** in typical sequences.

---

# Key Idea

Instead of committing to one model globally, the system **allocates compute selectively** based on detection confidence and scene context.

Heavy model escalation is triggered when the lightweight detector shows signs of failure:

1. **Low confidence detection**
2. **Detection streak break** (pedestrian disappears after multiple frames)
3. **Very small bounding boxes** (far-away pedestrians)

These signals capture common failure modes of lightweight detectors.

---

# Results

Evaluation was performed on **Caltech Pedestrian set00 (2,500 frames)** with IoU = 0.5.

| Configuration         | FPS       | Recall | Precision  | MD@100 |
| --------------------- | --------- | ------ | ---------- | ------ |
| YOLOv8n pretrained    | 124.5     | 0.2279 | 0.7671     | 143.28 |
| YOLOv8l pretrained    | 99.3      | 0.2774 | 0.7602     | 134.08 |
| YOLOv8n fine-tuned    | 127.9     | 0.2393 | 0.7317     | 141.16 |
| YOLOv8l fine-tuned    | 94.5      | 0.2645 | 0.8319     | 136.48 |
| **Adaptive Pipeline** | **124.6** | 0.2442 | **0.8655** | 140.24 |

The adaptive system achieves **higher precision than either standalone model**, while maintaining throughput close to the lightweight detector.

Recall falls between the two base models, which is expected for cascaded systems where both detectors share similar architecture.

---

# Precision vs Recall Tradeoff

The scheduler primarily improves **precision**, not recall.

Recall increases slightly relative to the lightweight model:

```
YOLOv8n fine-tuned   → 0.2393
Adaptive pipeline    → 0.2442
```

but does not reach the recall of the large model.

Instead, the scheduler improves **trustworthiness of detections** by replacing uncertain predictions with higher-quality outputs.

When the system emits a detection, it is more likely to be correct.

---

# Threshold Sweep

The threshold **T** controls escalation sensitivity.

Lower thresholds mean the system escalates **more conservatively**.

| T    | FPS   | Recall | MD@100 | Heavy Model Triggered |
| ---- | ----- | ------ | ------ | --------------------- |
| 0.25 | 129.0 | 0.2464 | 139.84 | 24.6%                 |
| 0.35 | 125.5 | 0.2442 | 140.24 | 27.0%                 |
| 0.45 | 124.2 | 0.2442 | 140.24 | 28.0%                 |
| 0.55 | 122.9 | 0.2442 | 140.24 | 28.8%                 |

The best operating point was **T = 0.25**, balancing recall, throughput, and compute cost.

---

# Domain Adaptation

Both models were pretrained on **COCO**, which contains relatively few dense urban pedestrian scenes.

Running pretrained models directly on driving footage shows a clear domain gap:

| Model              | Recall |
| ------------------ | ------ |
| YOLOv8n pretrained | 22.8%  |
| YOLOv8l pretrained | 27.7%  |

Fine-tuning on **CrowdHuman**, which contains dense and heavily occluded pedestrians, significantly improved performance:

| Model              | Recall | Precision | mAP50  |
| ------------------ | ------ | --------- | ------ |
| YOLOv8n fine-tuned | 0.7017 | 0.8487    | 0.8155 |
| YOLOv8l fine-tuned | 0.7922 | 0.8717    | 0.8792 |

This demonstrates how **dataset domain alignment** strongly affects detection quality.

---

# Scheduler Logic

```
Input Frame
     |
     v
YOLOv8n (every frame)
     |
     |-- All detections confident
     |      → use YOLOv8n result
     |
     |-- Any trigger fires:
     |      - detection confidence < T
     |      - detection streak breaks
     |      - bounding box < 1% image area
     |
     v
YOLOv8l
     |
     v
Final detection output
```

The scheduler captures three common failure modes:

| Trigger                | Failure Mode        |
| ---------------------- | ------------------- |
| Low confidence         | ambiguous detection |
| Detection streak break | missed pedestrian   |
| Small box              | distant pedestrian  |

---

# Why This Matters

Real-world perception systems operate under **strict compute budgets**.

The typical solution is to deploy a lightweight model and accept lower accuracy.

This project demonstrates an alternative strategy:

> **allocate compute dynamically where it matters most**

The system adjusts automatically to scene complexity:

* **Sparse scenes** → heavy model rarely used
* **Crowded scenes** → heavy model invoked more frequently

The policy is also **model-agnostic**. Any pair of fast and accurate models can be used.

---

# Datasets

### CrowdHuman

Used for model fine-tuning.

* 470k annotated persons
* average **23 people per image**
* heavy occlusion and crowding

### Caltech Pedestrian

Used for evaluation.

* vehicle-mounted urban driving footage
* 30 FPS video
* detailed occlusion annotations

Evaluating on a dataset different from training tests whether improvements generalize beyond the training distribution.

---

# Demo

![demo](results/figures/demo.gif)

---

# Project Structure

```
adaptive-pedestrian-detection/

data/
  caltech/
  crowdhuman/

src/
  scheduler.py

experiments/
  run_baseline.py
  run_finetuned_eval.py
  run_adaptive.py
  sweep_thresholds.py
  plot_results.py

results/
  full_comparison.csv
  sweep_results.csv
  figures/

demo/
  demo_video.py
```

---

# Tech Stack

Python 3.10
PyTorch
Ultralytics YOLOv8
OpenCV
NumPy
Pandas
Matplotlib

Hardware: **NVIDIA A100-SXM4-80GB**

---

# References

Shao et al. — *CrowdHuman: A Benchmark for Detecting Human in a Crowd*
Dollar et al. — *Pedestrian Detection: An Evaluation of the State of the Art*
Ultralytics — *YOLOv8*
