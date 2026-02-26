from ultralytics import YOLO
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nDownloading YOLOv8n...")
model_n = YOLO("yolov8n.pt")

print("\nDownloading YOLOv8l...")
model_l = YOLO("yolov8l.pt")

import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)

results_n = model_n(dummy, verbose=False)
results_l = model_l(dummy, verbose=False)

print(f"YOLOv8n — output shape: {results_n[0].boxes.shape}")
print(f"YOLOv8l — output shape: {results_l[0].boxes.shape}")