import time
import numpy as np
from ultralytics import YOLO


class AdaptiveScheduler:
    """
    Confidence-triggered cascaded inference scheduler.

    Every frame: run YOLOv8n (fast).
    Trigger YOLOv8l (heavy) if ANY of:
      1. Any detection has confidence < threshold T
      2. No pedestrian detected AND last N frames had detections (streak break)
      3. Any detected box is very small (far/hard pedestrian)

    YOLOv8l result replaces YOLOv8n result for triggered frames.
    """

    def __init__(
        self,
        light_weights: str,
        heavy_weights: str,
        conf_threshold: float = 0.35,
        small_box_thresh: float = 0.01,   
        streak_window: int = 3,           
        warmup_frames: int = 10,
    ):
        self.conf_threshold  = conf_threshold
        self.small_box_thresh = small_box_thresh
        self.streak_window   = streak_window

        print(f"Loading light model: {light_weights}")
        self.light_model = YOLO(light_weights)
        print(f"Loading heavy model: {heavy_weights}")
        self.heavy_model = YOLO(heavy_weights)

        self._recent_had_detection = []   

        self.stats = {
            "total_frames": 0,
            "heavy_triggered": 0,
            "light_only_frames": 0,
            "total_light_ms": 0.0,
            "total_heavy_ms": 0.0,
        }

        print(f"Warming up models ({warmup_frames} frames)...")
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(warmup_frames):
            self.light_model(dummy, verbose=False)
            self.heavy_model(dummy, verbose=False)
        print("Warmup done.\n")

    def _should_trigger_heavy(self, light_results, img_hw):
        """
        Decide whether to invoke the heavy model based on light model output.
        Returns (trigger: bool, reason: str)
        """
        H, W = img_hw
        img_area = H * W
        boxes = light_results[0].boxes

        had_detection = boxes is not None and len(boxes) > 0
        if not had_detection:
            recent_streak = sum(self._recent_had_detection[-self.streak_window:])
            if recent_streak >= self.streak_window:
                return True, "streak_break"

        if boxes is None or len(boxes) == 0:
            return False, "no_detection_no_streak"

        confs = boxes.conf.cpu().numpy()
        xyxy  = boxes.xyxy.cpu().numpy()

        if np.any(confs < self.conf_threshold):
            return True, f"low_conf({confs.min():.2f})"

        for box in xyxy:
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            if (box_area / img_area) < self.small_box_thresh:
                return True, f"small_box({box_area/img_area:.4f})"

        return False, "light_sufficient"

    def run_frame(self, frame_bgr):
        """
        Run the scheduler on a single BGR frame (numpy array).
        Returns:
            detections: list of {"bbox": [x,y,w,h], "conf": float, "model": "light"|"heavy"}
            triggered:  bool — whether heavy model was invoked
            reason:     str  — trigger reason
            fps:        float — effective FPS for this frame
        """
        H, W = frame_bgr.shape[:2]
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        light_results = self.light_model(frame_bgr, classes=[0], verbose=False)
        light_ms = (time.perf_counter() - t0) * 1000
        self.stats["total_light_ms"] += light_ms

        triggered, reason = self._should_trigger_heavy(light_results, (H, W))

        if triggered:
            t0 = time.perf_counter()
            final_results = self.heavy_model(frame_bgr, classes=[0], verbose=False)
            heavy_ms = (time.perf_counter() - t0) * 1000
            self.stats["total_heavy_ms"] += heavy_ms
            self.stats["heavy_triggered"] += 1
            used_model = "heavy"
        else:
            final_results = light_results
            self.stats["light_only_frames"] += 1
            used_model = "light"

        had_det = (final_results[0].boxes is not None and len(final_results[0].boxes) > 0)
        self._recent_had_detection.append(had_det)
        if len(self._recent_had_detection) > self.streak_window + 2:
            self._recent_had_detection.pop(0)

        self.stats["total_frames"] += 1

        detections = []
        if final_results[0].boxes is not None:
            xyxy  = final_results[0].boxes.xyxy.cpu().numpy()
            confs = final_results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = box
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "conf": float(conf),
                    "model": used_model,
                })

        total_ms = (time.perf_counter() - t_start) * 1000
        fps = 1000 / total_ms if total_ms > 0 else 0.0

        return detections, triggered, reason, fps

    def summary(self):
        s = self.stats
        n = s["total_frames"]
        if n == 0:
            return {}
        pct_heavy = 100 * s["heavy_triggered"] / n
        avg_light_ms = s["total_light_ms"] / n
        avg_heavy_ms = s["total_heavy_ms"] / s["heavy_triggered"] if s["heavy_triggered"] > 0 else 0
        effective_ms = (s["total_light_ms"] + s["total_heavy_ms"]) / n
        effective_fps = 1000 / effective_ms if effective_ms > 0 else 0

        return {
            "total_frames": n,
            "heavy_triggered": s["heavy_triggered"],
            "pct_heavy": round(pct_heavy, 1),
            "avg_light_ms": round(avg_light_ms, 2),
            "avg_heavy_ms": round(avg_heavy_ms, 2),
            "effective_fps": round(effective_fps, 1),
        }