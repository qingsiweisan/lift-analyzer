"""YOLOv8-based barbell detection for accurate bar tracking.

Replaces wrist-based approximation with direct barbell bounding box detection.
Falls back to wrist tracking when barbell is not detected.

Requires: ultralytics (pip install ultralytics)
"""

import os
import numpy as np

_model = None
_model_path = None


def _get_model(model_path=None):
    """Lazy-load YOLOv8 model.

    Args:
        model_path: Path to a custom-trained barbell detection model (.pt).
                    If None, uses the default model path from BARBELL_MODEL_PATH
                    env var, or falls back to a bundled model.
    """
    global _model, _model_path

    path = model_path or os.environ.get("BARBELL_MODEL_PATH")
    if path and path == _model_path and _model is not None:
        return _model

    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "ultralytics not installed. Run: pip install ultralytics\n"
            "For barbell detection, you also need a trained model."
        )

    if path and os.path.isfile(path):
        _model = YOLO(path)
        _model_path = path
    else:
        # Use YOLOv8n as base — user should provide a fine-tuned model
        # for barbell detection. Without a custom model, we use the COCO
        # pretrained model which can detect "sports ball" but not barbells.
        _model = YOLO("yolov8n.pt")
        _model_path = "yolov8n.pt"
        print("  [barbell] Using default YOLOv8n (COCO). For better results,")
        print("            set BARBELL_MODEL_PATH to a barbell-trained model.")

    return _model


class BarbellTracker:
    """Track barbell position across video frames using YOLOv8.

    Falls back to wrist position when detection fails.
    """

    # COCO class IDs that might be a barbell (for pretrained model)
    # In practice, a custom model with class 0 = "barbell" is recommended
    BARBELL_CLASSES = None  # Set after model loads

    def __init__(self, model_path=None, conf_threshold=0.3):
        """
        Args:
            model_path: Path to YOLOv8 barbell model. None = default.
            conf_threshold: Minimum confidence for detection.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self._model = None
        self._class_names = {}
        self._barbell_class_ids = set()
        self._history = []  # (x, y) history for smoothing
        self._fallback_count = 0
        self._detect_count = 0

    def _ensure_model(self):
        if self._model is not None:
            return
        self._model = _get_model(self.model_path)
        # Discover class names
        if hasattr(self._model, "names"):
            self._class_names = self._model.names
            # Find barbell-like classes
            for idx, name in self._class_names.items():
                name_lower = name.lower()
                if any(kw in name_lower for kw in ["barbell", "bar", "dumbbell", "weight"]):
                    self._barbell_class_ids.add(idx)
            # If no barbell class found (COCO model), we'll rely on spatial heuristics
            if not self._barbell_class_ids:
                self._barbell_class_ids = None

    def detect(self, frame, wrist_fallback=None):
        """Detect barbell in a single frame.

        Args:
            frame: BGR numpy array.
            wrist_fallback: (x, y) wrist position to use if detection fails.

        Returns:
            dict with:
                "x": center x in pixels
                "y": center y in pixels
                "confidence": detection confidence (0 if fallback)
                "source": "yolo" or "wrist"
                "bbox": (x1, y1, x2, y2) or None
        """
        self._ensure_model()

        results = self._model(frame, verbose=False, conf=self.conf_threshold)

        best_box = None
        best_conf = 0

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # If we have barbell-specific classes, filter by them
                if self._barbell_class_ids is not None:
                    if cls_id not in self._barbell_class_ids:
                        continue
                else:
                    # COCO fallback: look for any detection near expected bar position
                    # Skip person detections (class 0 in COCO)
                    if cls_id == 0:
                        continue

                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].cpu().numpy()

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            self._history.append((cx, cy))
            self._detect_count += 1
            return {
                "x": cx, "y": cy,
                "confidence": best_conf,
                "source": "yolo",
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
            }

        # Fallback to wrist
        self._fallback_count += 1
        if wrist_fallback:
            self._history.append(wrist_fallback)
            return {
                "x": wrist_fallback[0], "y": wrist_fallback[1],
                "confidence": 0.0,
                "source": "wrist",
                "bbox": None,
            }

        return {
            "x": 0, "y": 0,
            "confidence": 0.0,
            "source": "none",
            "bbox": None,
        }

    def get_stats(self):
        """Return detection statistics."""
        total = self._detect_count + self._fallback_count
        return {
            "total_frames": total,
            "yolo_detections": self._detect_count,
            "wrist_fallbacks": self._fallback_count,
            "detection_rate": self._detect_count / total if total > 0 else 0,
        }

    def reset(self):
        """Reset tracker state for a new video."""
        self._history = []
        self._fallback_count = 0
        self._detect_count = 0
