from ultralytics import YOLO
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nDownloading YOLOv8n...")
model_n = YOLO("yolov8n.pt")
print("YOLOv8n loaded OK")

print("\nDownloading YOLOv8l...")
model_l = YOLO("yolov8l.pt")
print("YOLOv8l loaded OK")

print("\nRunning quick inference test on both models...")
import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)

results_n = model_n(dummy, verbose=False)
results_l = model_l(dummy, verbose=False)

print(f"YOLOv8n inference OK — output shape: {results_n[0].boxes.shape}")
print(f"YOLOv8l inference OK — output shape: {results_l[0].boxes.shape}")

print("\nAll good. Models ready.")