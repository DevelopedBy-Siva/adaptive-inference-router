import sys, os
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from src.scheduler import AdaptiveScheduler
from data.caltech.parse_annotations import load_annotations_for_sequences

LIGHT = "runs/detect/models/finetuned/yolov8n_finetuned/weights/best.pt"
HEAVY = "runs/detect/models/finetuned/yolov8l_finetuned/weights/best.pt"

CALTECH_ROOT = "data/caltech"
FRAMES_ROOT  = "data/caltech/frames"
RESULTS_PATH = "results/sweep_results.csv"
IOU_THRESH   = 0.5
MAX_FRAMES   = 500

SEQUENCES = [
    ("set00", "V000"),
    ("set00", "V001"),
    ("set00", "V002"),
    ("set00", "V003"),
    ("set00", "V004"),
]

THRESHOLDS = [0.25, 0.35, 0.45, 0.55]


def compute_iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0]+boxA[2], boxA[1]+boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0]+boxB[2], boxB[1]+boxB[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)


def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    if len(gt_boxes) == 0: return 0, 0, len(pred_boxes)
    if len(pred_boxes) == 0: return 0, len(gt_boxes), 0
    matched = set()
    tp = 0
    for gt in gt_boxes:
        best_iou, best_j = 0, -1
        for j, pred in enumerate(pred_boxes):
            if j in matched: continue
            iou = compute_iou(gt, pred)
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(best_j)
    return tp, len(gt_boxes)-tp, len(pred_boxes)-len(matched)


def get_frame_paths(set_name, vid_name):
    return sorted((Path(FRAMES_ROOT)/set_name/vid_name).glob("frame_*.jpg"))[:MAX_FRAMES]


def eval_threshold(T, all_annotations):
    print(f"\n{'='*55}")
    print(f"  Threshold T = {T}")
    print(f"{'='*55}")

    scheduler = AdaptiveScheduler(
        light_weights=LIGHT,
        heavy_weights=HEAVY,
        conf_threshold=T,
        warmup_frames=10,
    )

    total_frames = total_tp = total_fn = total_fp = 0

    for (set_name, vid_name), frame_anns in all_annotations.items():
        frame_paths = get_frame_paths(set_name, vid_name)
        print(f"  {set_name}/{vid_name}: {len(frame_paths)} frames")

        for fp in frame_paths:
            frame_idx = int(fp.stem.split("_")[1])
            gt_anns  = frame_anns.get(frame_idx, [])
            gt_boxes = [a["bbox"] for a in gt_anns if a["label"] in ["person","people"]]

            frame_bgr = cv2.imread(str(fp))
            detections, _, _, _ = scheduler.run_frame(frame_bgr)
            pred_boxes = [d["bbox"] for d in detections]

            tp, fn, fp_count = match_detections(gt_boxes, pred_boxes, IOU_THRESH)
            total_tp += tp
            total_fn += fn
            total_fp += fp_count
            total_frames += 1

    summary = scheduler.summary()
    recall    = total_tp / (total_tp + total_fn) if (total_tp+total_fn) > 0 else 0.0
    md_100    = (total_fn / total_frames) * 100 if total_frames > 0 else 0.0

    row = {
        "threshold": T,
        "fps": summary["effective_fps"],
        "recall": round(recall, 4),
        "md_100": round(md_100, 2),
        "pct_heavy": summary["pct_heavy"],
    }

    print(f"\n  T={T} → FPS={row['fps']} | Recall={row['recall']} | MD@100={row['md_100']} | Heavy%={row['pct_heavy']}%")
    return row


if __name__ == "__main__":
    print("Loading annotations...")
    all_annotations = load_annotations_for_sequences(CALTECH_ROOT, SEQUENCES)

    rows = []
    for T in THRESHOLDS:
        rows.append(eval_threshold(T, all_annotations))

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)

    print(f"\n\n{'='*55}")
    print("── Threshold Sweep Results ──")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_PATH}")

    viable = df[df["fps"] >= 100]
    if len(viable) > 0:
        best = viable.loc[viable["recall"].idxmax()]
        print(f"\nBest tradeoff (FPS≥100): T={best['threshold']} → FPS={best['fps']}, Recall={best['recall']}, MD@100={best['md_100']}, Heavy%={best['pct_heavy']}%")
    else:
        best = df.loc[df["recall"].idxmax()]
        print(f"\nNo config hits FPS≥100. Best recall: T={best['threshold']} → FPS={best['fps']}, Recall={best['recall']}")