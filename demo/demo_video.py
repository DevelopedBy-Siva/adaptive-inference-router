import sys, os
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from pathlib import Path
from src.scheduler import AdaptiveScheduler

LIGHT = "runs/detect/models/finetuned/yolov8n_finetuned/weights/best.pt"
HEAVY = "runs/detect/models/finetuned/yolov8l_finetuned/weights/best.pt"

FRAMES_DIR = "data/caltech/frames/set00/V001"
OUTPUT_PATH = "demo/adaptive_demo.mp4"
CONF_T      = 0.25
TARGET_FPS  = 30
MAX_FRAMES  = 900  

COLOR_LIGHT  = (50, 200, 50)    
COLOR_HEAVY  = (50, 50, 220)   
COLOR_BOX    = (0, 220, 255)   
COLOR_TEXT   = (255, 255, 255)
COLOR_BG     = (0, 0, 0)


def draw_overlay(frame, detections, triggered, reason, fps, frame_idx, stats):
    H, W = frame.shape[:2]
    out = frame.copy()

    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        color = COLOR_HEAVY if det["model"] == "heavy" else COLOR_LIGHT
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        conf_txt = f"{det['conf']:.2f}"
        cv2.putText(out, conf_txt, (x, max(y-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (W, 52), (20, 20, 20), -1)

    model_label = "HEAVY MODEL" if triggered else "light model"
    model_color = COLOR_HEAVY if triggered else COLOR_LIGHT
    cv2.putText(out, model_label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, model_color, 2, cv2.LINE_AA)

    cv2.putText(out, f"FPS: {fps:.1f}", (W - 160, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.putText(out, f"Frame: {frame_idx}", (W - 160, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.putText(out, f"trigger: {reason}", (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, H-30), (W, H), (20, 20, 20), -1)
    pct = 100 * stats["heavy"] / stats["total"] if stats["total"] > 0 else 0
    stats_txt = f"Heavy invoked: {stats['heavy']}/{stats['total']} frames ({pct:.1f}%)  |  T={CONF_T}"
    cv2.putText(out, stats_txt, (10, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    return out


if __name__ == "__main__":
    os.makedirs("demo", exist_ok=True)

    scheduler = AdaptiveScheduler(
        light_weights=LIGHT,
        heavy_weights=HEAVY,
        conf_threshold=CONF_T,
        warmup_frames=10,
    )

    frame_paths = sorted(Path(FRAMES_DIR).glob("frame_*.jpg"))[:MAX_FRAMES]
    print(f"Processing {len(frame_paths)} frames from {FRAMES_DIR}")

    sample = cv2.imread(str(frame_paths[0]))
    H, W = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, TARGET_FPS, (W, H))

    stats = {"total": 0, "heavy": 0}

    for i, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue

        detections, triggered, reason, fps = scheduler.run_frame(frame)

        stats["total"] += 1
        if triggered:
            stats["heavy"] += 1

        out_frame = draw_overlay(frame, detections, triggered, reason, fps, i, stats)
        writer.write(out_frame)

        if i % 100 == 0:
            pct = 100*stats["heavy"]/stats["total"] if stats["total"]>0 else 0
            print(f"  Frame {i:4d}/{len(frame_paths)} | Heavy: {pct:.1f}% | Last: {'HEAVY' if triggered else 'light'} ({reason})")

    writer.release()
    summary = scheduler.summary()

    print(f"\nDemo video saved to {OUTPUT_PATH}")
    print(f"   Duration: {len(frame_paths)/TARGET_FPS:.1f}s at {TARGET_FPS}fps")
    print(f"   Heavy triggered: {summary['pct_heavy']}% of frames")
    print(f"   Effective FPS: {summary['effective_fps']}")

    if summary["pct_heavy"] > 0 and summary["pct_heavy"] < 100:
        print("\nPASS: Light/heavy switching is visible in demo video.")
    elif summary["pct_heavy"] == 0:
        print("\nFAIL: Heavy model never triggered — lower threshold T or check trigger logic.")
    else:
        print("\nFAIL: Heavy model triggered every frame — raise threshold T.")