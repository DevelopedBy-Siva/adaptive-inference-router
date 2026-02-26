# src/finetune.py
import os
import sys
from pathlib import Path
from ultralytics import YOLO

# ── Config ───────────────────────────────────────────────────────────────────
YAML_PATH   = "data/crowdhuman_yolo/crowdhuman.yaml"
OUTPUT_DIR  = "models/finetuned"
EPOCHS      = 30          # enough for convergence on CrowdHuman
IMGSZ       = 640
BATCH       = 32          # A100 80GB can handle this for both models
WORKERS     = 8
DEVICE      = 0           # GPU 0

MODELS_TO_FINETUNE = [
    ("yolov8n_finetuned", "yolov8n.pt"),
    ("yolov8l_finetuned", "yolov8l.pt"),
]
# ─────────────────────────────────────────────────────────────────────────────


def finetune(run_name, base_weights):
    print(f"\n{'='*60}")
    print(f"Fine-tuning: {run_name}")
    print(f"Base weights: {base_weights}")
    print(f"Epochs: {EPOCHS} | Batch: {BATCH} | ImgSz: {IMGSZ}")
    print(f"{'='*60}\n")

    model = YOLO(base_weights)

    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        name=run_name,
        project=OUTPUT_DIR,
        patience=10,           # early stopping if no improvement for 10 epochs
        save=True,
        save_period=5,         # checkpoint every 5 epochs
        val=True,
        plots=True,
        # Training hyperparameters — keep identical for fair comparison
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentation — standard for pedestrian detection
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        verbose=True,
    )

    best_weights = Path(OUTPUT_DIR) / run_name / "weights" / "best.pt"
    print(f"\n✅ {run_name} done.")
    print(f"   Best weights: {best_weights}")
    print(f"   Final metrics from training:")
    print(f"   metrics/recall(B)    = {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
    print(f"   metrics/precision(B) = {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"   metrics/mAP50(B)     = {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

    return str(best_weights)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trained_paths = {}
    for run_name, base_weights in MODELS_TO_FINETUNE:
        weights_path = finetune(run_name, base_weights)
        trained_paths[run_name] = weights_path

    print("\n── All fine-tuning complete ──")
    for name, path in trained_paths.items():
        exists = Path(path).exists()
        print(f"  {name}: {path} — {'✅ exists' if exists else '❌ MISSING'}")