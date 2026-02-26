import sys
import os
sys.path.insert(0, os.path.abspath("."))

import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from data.caltech.parse_annotations import load_annotations_for_sequences

CALTECH_ROOT = "data/caltech"
FRAMES_ROOT  = "data/caltech/frames"
RESULTS_PATH = "results/baseline_metrics.csv"
IOU_THRESH   = 0.5
MAX_FRAMES   = 500  

SEQUENCES = [
    ("set00", "V000"),
    ("set00", "V001"),
    ("set00", "V002"),
    ("set00", "V003"),
    ("set00", "V004"),
]

MODELS = {
    "yolov8n_pretrained": "yolov8n.pt",
    "yolov8l_pretrained": "yolov8l.pt",
}


def compute_iou(boxA, boxB):
    """boxA, boxB: [x, y, w, h] format"""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    return inter / (areaA + areaB - inter)


def match_detections(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    Greedy matching: for each GT box, find best unmatched pred box.
    Returns: (tp, fn, fp)
    """
    if len(gt_boxes) == 0:
        return 0, 0, len(pred_boxes)
    if len(pred_boxes) == 0:
        return 0, len(gt_boxes), 0

    matched_pred = set()
    tp = 0
    for gt in gt_boxes:
        best_iou = 0
        best_j = -1
        for j, pred in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh:
            tp += 1
            matched_pred.add(best_j)

    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(matched_pred)
    return tp, fn, fp


def get_frame_paths(set_name, vid_name, max_frames):
    frame_dir = Path(FRAMES_ROOT) / set_name / vid_name
    paths = sorted(frame_dir.glob("frame_*.jpg"))[:max_frames]
    return paths


def run_model_on_sequences(model_name, model_path, all_annotations):
    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"{'='*60}")

    model = YOLO(model_path)

    total_frames = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0

    occl_tp = {0: 0, 1: 0, 2: 0}
    occl_fn = {0: 0, 1: 0, 2: 0}

    inference_times = []

    for (set_name, vid_name), frame_anns in all_annotations.items():
        frame_paths = get_frame_paths(set_name, vid_name, MAX_FRAMES)
        print(f"\n  {set_name}/{vid_name}: {len(frame_paths)} frames")

        for frame_path in frame_paths:
            frame_idx = int(frame_path.stem.split("_")[1])
            gt_anns = frame_anns.get(frame_idx, [])

            gt_boxes = [a["bbox"] for a in gt_anns if a["label"] in ["person", "people"]]
            gt_occls = [a["occl"] for a in gt_anns if a["label"] in ["person", "people"]]

            t0 = time.perf_counter()
            results = model(str(frame_path), classes=[0], verbose=False)  # class 0 = person in COCO
            t1 = time.perf_counter()
            inference_times.append(t1 - t0)

            pred_boxes = []
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                xyxy = results[0].boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    x1, y1, x2, y2 = box
                    pred_boxes.append([x1, y1, x2 - x1, y2 - y1])

            tp, fn, fp = match_detections(gt_boxes, pred_boxes, IOU_THRESH)
            total_tp += tp
            total_fn += fn
            total_fp += fp
            total_frames += 1

            for i, gt_box in enumerate(gt_boxes):
                occl = gt_occls[i] if i < len(gt_occls) else 0
                occl = min(occl, 2)  # cap at 2
                tp_i, fn_i, _ = match_detections([gt_box], pred_boxes, IOU_THRESH)
                occl_tp[occl] += tp_i
                occl_fn[occl] += fn_i

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    md_100 = (total_fn / total_frames) * 100 if total_frames > 0 else 0.0
    avg_ms = np.mean(inference_times) * 1000
    fps = 1000 / avg_ms if avg_ms > 0 else 0.0

    def miss_rate(occl_level):
        denom = occl_tp[occl_level] + occl_fn[occl_level]
        if denom == 0:
            return float('nan')
        return occl_fn[occl_level] / denom

    mr_none    = miss_rate(0)
    mr_partial = miss_rate(1)
    mr_heavy   = miss_rate(2)

    print(f"\n  ── Results for {model_name} ──")
    print(f"  Frames evaluated : {total_frames}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  MD@100           : {md_100:.2f}")
    print(f"  Avg FPS          : {fps:.1f}")
    print(f"  Miss rate (none) : {mr_none:.4f}")
    print(f"  Miss rate (part) : {mr_partial:.4f}")
    print(f"  Miss rate (heavy): {mr_heavy:.4f}")

    return {
        "model": model_name,
        "frames": total_frames,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "md_100": round(md_100, 2),
        "fps": round(fps, 1),
        "miss_rate_none": round(mr_none, 4) if not np.isnan(mr_none) else None,
        "miss_rate_partial": round(mr_partial, 4) if not np.isnan(mr_partial) else None,
        "miss_rate_heavy": round(mr_heavy, 4) if not np.isnan(mr_heavy) else None,
    }


if __name__ == "__main__":
    print("Loading annotations...")
    all_annotations = load_annotations_for_sequences(CALTECH_ROOT, SEQUENCES)

    results_list = []
    for model_name, model_path in MODELS.items():
        row = run_model_on_sequences(model_name, model_path, all_annotations)
        results_list.append(row)

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results_list)
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(df.to_string(index=False))

    print("\n── Sanity Check ──")
    n_row = df[df.model == "yolov8n_pretrained"].iloc[0]
    l_row = df[df.model == "yolov8l_pretrained"].iloc[0]
    if l_row.recall > n_row.recall:
        print("PASS: YOLOv8l recall > YOLOv8n recall — expected")
    else:
        print("FAIL: YOLOv8l recall <= YOLOv8n recall — something is wrong, debug before proceeding")
    if n_row.fps > l_row.fps:
        print("PASS: YOLOv8n FPS > YOLOv8l FPS — expected")
    else:
        print("FAIL: YOLOv8n FPS <= YOLOv8l FPS — check inference setup")