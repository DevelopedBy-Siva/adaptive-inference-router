# experiments/validate_scheduler.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import cv2
from pathlib import Path
from src.scheduler import AdaptiveScheduler

LIGHT = "runs/detect/models/finetuned/yolov8n_finetuned/weights/best.pt"
HEAVY = "runs/detect/models/finetuned/yolov8l_finetuned/weights/best.pt"
FRAMES_DIR = "data/caltech/frames/set00/V001"
N_FRAMES = 200
THRESHOLD = 0.35

scheduler = AdaptiveScheduler(
    light_weights=LIGHT,
    heavy_weights=HEAVY,
    conf_threshold=THRESHOLD,
    warmup_frames=10,
)

frame_paths = sorted(Path(FRAMES_DIR).glob("frame_*.jpg"))[:N_FRAMES]
print(f"Running scheduler on {len(frame_paths)} frames (T={THRESHOLD})...\n")

trigger_log = []
for i, fp in enumerate(frame_paths):
    frame = cv2.imread(str(fp))
    detections, triggered, reason, fps = scheduler.run_frame(frame)
    trigger_log.append((triggered, reason))
    if i % 50 == 0:
        print(f"  Frame {i:3d}: {'HEAVY' if triggered else 'light':5s} | {reason:30s} | {len(detections)} dets | {fps:.1f} FPS")

summary = scheduler.summary()
print(f"\n── Scheduler Summary (T={THRESHOLD}) ──")
for k, v in summary.items():
    print(f"  {k:20s}: {v}")

pct = summary["pct_heavy"]
print(f"\n── Sanity Check ──")
if 5 <= pct <= 80:
    print(f"✅ PASS: {pct}% heavy triggers — reasonable range (5–80%)")
    print("   Scheduler is working. Proceed to threshold sweep.")
elif pct == 0:
    print(f"❌ FAIL: 0% heavy triggers — threshold T={THRESHOLD} is too high, lower it")
elif pct > 80:
    print(f"❌ FAIL: {pct}% heavy triggers — threshold T={THRESHOLD} is too low, raise it")
else:
    print(f"⚠️  Borderline: {pct}% — check trigger reasons above")