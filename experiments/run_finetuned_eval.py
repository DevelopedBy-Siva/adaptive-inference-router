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
EXISTING_BASELINE_CSV = "results/baseline_metrics.csv"
RESULTS_PATH = "results/finetuned_metrics.csv"
FULL_TABLE_PATH = "results/full_comparison.csv"
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
    "yolov8n_finetuned": "runs/detect/models/finetuned/yolov8n_finetuned/weights/best.pt",
    "yolov8l_finetuned": "runs/detect/models/finetuned/yolov8l_finetuned/weights/best.pt",
}


def compute_iou(boxA, boxB):
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
    if len(gt_boxes) == 0:
        return 0, 0, len(pred_boxes)
    if len(pred_boxes) == 0:
        return 0, len(gt_boxes), 0
    matched_pred = set()
    tp = 0
    for gt in gt_boxes:
        best_iou, best_j = 0, -1
        for j, pred in enumerate(pred_boxes):
            if j in matched_pred:
                continue
            iou = compute_iou(gt, pred)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched_pred.add(best_j)
    fn = len(gt_boxes) - tp
    fp = len(pred_boxes) - len(matched_pred)
    return tp, fn, fp


def get_frame_paths(set_name, vid_name, max_frames):
    frame_dir = Path(FRAMES_ROOT) / set_name / vid_name
    return sorted(frame_dir.glob("frame_*.jpg"))[:max_frames]


def run_model_on_sequences(model_name, model_path, all_annotations):
    print(f"\n{'='*60}")
    print(f"Running: {model_name}")
    print(f"Weights: {model_path}")
    print(f"{'='*60}")

    if not Path(model_path).exists():
        print(f"ERROR: weights not found at {model_path}")
        return None

    model = YOLO(model_path)

    print("  Warming up model...")
    warmup_img = str(get_frame_paths(SEQUENCES[0][0], SEQUENCES[0][1], 1)[0])
    for _ in range(10):
        model(warmup_img, verbose=False)
    print("  Warmup done.")

    total_frames = 0
    total_tp = total_fn = total_fp = 0
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
            results = model(str(frame_path), classes=[0], verbose=False)
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
                occl = min(gt_occls[i] if i < len(gt_occls) else 0, 2)
                tp_i, fn_i, _ = match_detections([gt_box], pred_boxes, IOU_THRESH)
                occl_tp[occl] += tp_i
                occl_fn[occl] += fn_i

    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    md_100    = (total_fn / total_frames) * 100 if total_frames > 0 else 0.0
    avg_ms    = np.mean(inference_times) * 1000
    fps       = 1000 / avg_ms if avg_ms > 0 else 0.0

    def miss_rate(o):
        d = occl_tp[o] + occl_fn[o]
        return occl_fn[o] / d if d > 0 else float('nan')

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
    print(f"  Miss rate (heavy): {mr_heavy:.4f}" if not np.isnan(mr_heavy) else "  Miss rate (heavy): N/A (no heavily occluded GT in set00)")

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
        if row:
            results_list.append(row)

    os.makedirs("results", exist_ok=True)
    df_new = pd.DataFrame(results_list)
    df_new.to_csv(RESULTS_PATH, index=False)
    print(f"\nFine-tuned results saved to {RESULTS_PATH}")

    df_base = pd.read_csv(EXISTING_BASELINE_CSV)
    df_full = pd.concat([df_base, df_new], ignore_index=True)
    df_full.to_csv(FULL_TABLE_PATH, index=False)

    print(f"\n── Full Comparison Table ──")
    print(df_full[["model", "fps", "recall", "md_100", "miss_rate_partial"]].to_string(index=False))

    print("\n── Sanity Checks ──")
    rows = {r["model"]: r for r in df_full.to_dict("records")}

    checks = [
        ("yolov8l_finetuned", "yolov8n_finetuned", "recall",
         "YOLOv8l finetuned recall > YOLOv8n finetuned"),
        ("yolov8n_finetuned", "yolov8n_pretrained", "recall",
         "YOLOv8n finetuned recall > YOLOv8n pretrained (fine-tuning helped n)"),
        ("yolov8n_finetuned", "yolov8l_finetuned", "fps",
         "YOLOv8n finetuned FPS > YOLOv8l finetuned (still faster after warmup)"),
    ]

    all_pass = True
    for m1, m2, metric, desc in checks:
        if m1 not in rows or m2 not in rows:
            print(f"   SKIP: {desc} — model not found")
            continue
        v1, v2 = rows[m1][metric], rows[m2][metric]
        if v1 > v2:
            print(f"  PASS: {desc} ({v1} > {v2})")
        else:
            print(f"  FAIL: {desc} ({v1} <= {v2}) — investigate before Day 6")
            all_pass = False

    if all_pass:
        print("\nAll sanity checks passed. Ready for Day 6: Scheduler implementation.")
    else:
        print("\nSome checks failed. Debug before building the scheduler.")