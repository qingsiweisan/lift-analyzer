"""MediaPipe Pose wrapper for landmark extraction.

Supports both the legacy `mp.solutions.pose` API and the new Tasks API
(mediapipe >= 0.10.22 on Python 3.13+).
"""

import os
import cv2
import numpy as np

# Detect which MediaPipe API is available
_USE_TASKS_API = False
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    _USE_TASKS_API = True
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, PoseLandmark, RunningMode,
        PoseLandmarksConnections,
    )
    from mediapipe.tasks.python import BaseOptions
    import mediapipe as mp


# Model path for Tasks API
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_heavy.task")


if _USE_TASKS_API:
    # Tasks API landmark indices
    LANDMARKS_LEFT = {
        "ear": PoseLandmark.LEFT_EAR,
        "shoulder": PoseLandmark.LEFT_SHOULDER,
        "elbow": PoseLandmark.LEFT_ELBOW,
        "wrist": PoseLandmark.LEFT_WRIST,
        "hip": PoseLandmark.LEFT_HIP,
        "knee": PoseLandmark.LEFT_KNEE,
        "ankle": PoseLandmark.LEFT_ANKLE,
        "heel": PoseLandmark.LEFT_HEEL,
        "foot_index": PoseLandmark.LEFT_FOOT_INDEX,
    }
    LANDMARKS_RIGHT = {
        "ear": PoseLandmark.RIGHT_EAR,
        "shoulder": PoseLandmark.RIGHT_SHOULDER,
        "elbow": PoseLandmark.RIGHT_ELBOW,
        "wrist": PoseLandmark.RIGHT_WRIST,
        "hip": PoseLandmark.RIGHT_HIP,
        "knee": PoseLandmark.RIGHT_KNEE,
        "ankle": PoseLandmark.RIGHT_ANKLE,
        "heel": PoseLandmark.RIGHT_HEEL,
        "foot_index": PoseLandmark.RIGHT_FOOT_INDEX,
    }
else:
    # Legacy solutions API
    LANDMARKS_LEFT = {
        "ear": mp_pose.PoseLandmark.LEFT_EAR,
        "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "hip": mp_pose.PoseLandmark.LEFT_HIP,
        "knee": mp_pose.PoseLandmark.LEFT_KNEE,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "heel": mp_pose.PoseLandmark.LEFT_HEEL,
        "foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    }
    LANDMARKS_RIGHT = {
        "ear": mp_pose.PoseLandmark.RIGHT_EAR,
        "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
        "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "hip": mp_pose.PoseLandmark.RIGHT_HIP,
        "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
        "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
        "heel": mp_pose.PoseLandmark.RIGHT_HEEL,
        "foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
    }


class PoseWrapper:
    """Unified wrapper for both legacy and Tasks API pose estimation."""

    def __init__(self, model_complexity=2, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        if _USE_TASKS_API:
            if not os.path.isfile(_MODEL_PATH):
                raise FileNotFoundError(
                    f"Pose model not found: {_MODEL_PATH}\n"
                    "Download it:\n"
                    "  curl -L -o models/pose_landmarker_heavy.task "
                    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
                )
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._timestamp_ms = 0
        else:
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._landmarker = None

    def process(self, rgb_frame):
        """Process an RGB frame and return a PoseResult.

        Returns:
            PoseResult with .landmarks (list of landmarks) or None if no pose detected.
        """
        if _USE_TASKS_API:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._timestamp_ms += 33  # ~30fps increment
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                return _TasksResult(result.pose_landmarks[0])
            return None
        else:
            result = self._pose.process(rgb_frame)
            if result.pose_landmarks:
                return _LegacyResult(result.pose_landmarks)
            return None

    def close(self):
        if _USE_TASKS_API:
            self._landmarker.close()
        else:
            self._pose.close()


class _TasksResult:
    """Adapter for Tasks API result to match legacy API interface."""

    def __init__(self, landmarks):
        # landmarks is a list of NormalizedLandmark objects
        self._landmarks = landmarks
        self.pose_landmarks = self  # for draw_pose compat

    @property
    def landmark(self):
        return self._landmarks


class _LegacyResult:
    """Adapter for legacy API result."""

    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks

    @property
    def landmark(self):
        return self.pose_landmarks.landmark


def create_pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create a pose estimator (auto-selects API version)."""
    return PoseWrapper(model_complexity, min_detection_confidence, min_tracking_confidence)


def get_coords(landmarks, idx, width, height):
    """Get (x, y) pixel coordinates for a landmark index.

    Works with both legacy (landmark[idx]) and Tasks API (landmark[idx]).
    """
    if isinstance(idx, int):
        lm = landmarks[idx]
    else:
        # Enum value
        lm = landmarks[int(idx)]
    return (lm.x * width, lm.y * height), lm.visibility


def pick_visible_side(landmarks):
    """Pick the side (left/right) with higher hip visibility."""
    left_hip_idx = int(LANDMARKS_LEFT["hip"])
    right_hip_idx = int(LANDMARKS_RIGHT["hip"])
    left_vis = landmarks[left_hip_idx].visibility
    right_vis = landmarks[right_hip_idx].visibility
    if left_vis >= right_vis:
        return "left", LANDMARKS_LEFT
    return "right", LANDMARKS_RIGHT


def extract_points(landmarks, side_landmarks, width, height):
    """Extract all named points for a given side."""
    points = {}
    for name, idx in side_landmarks.items():
        coords, _ = get_coords(landmarks, idx, width, height)
        points[name] = coords
    return points


def extract_both_sides(landmarks, width, height):
    """Extract points for both left and right sides."""
    return {
        "left": extract_points(landmarks, LANDMARKS_LEFT, width, height),
        "right": extract_points(landmarks, LANDMARKS_RIGHT, width, height),
    }


def draw_pose(frame, pose_result):
    """Draw pose landmarks and connections on a frame."""
    if _USE_TASKS_API:
        _draw_pose_tasks(frame, pose_result)
    else:
        if not _USE_TASKS_API:
            mp_drawing.draw_landmarks(
                frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
            )


def _draw_pose_tasks(frame, pose_result):
    """Draw pose for Tasks API using OpenCV directly."""
    landmarks = pose_result.landmark if hasattr(pose_result, 'landmark') else pose_result._landmarks
    h, w = frame.shape[:2]

    # Draw landmarks as green dots
    coords = []
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        coords.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Draw connections
    _CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
        (11, 23), (12, 24), (23, 24),  # torso
        (23, 25), (25, 27), (24, 26), (26, 28),  # legs
        (27, 29), (29, 31), (28, 30), (30, 32),  # feet
    ]
    for i, j in _CONNECTIONS:
        if i < len(coords) and j < len(coords):
            cv2.line(frame, coords[i], coords[j], (0, 200, 255), 2)
