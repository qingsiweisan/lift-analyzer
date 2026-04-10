"""Video I/O and frame sampling utilities."""

import os
import cv2


class VideoReader:
    """Wraps cv2.VideoCapture with metadata."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def frames(self):
        """Yield (frame_idx, bgr_frame) for each frame."""
        idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield idx, frame
            idx += 1

    def close(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def info_str(self):
        return (f"Video: {self.width}x{self.height}, {self.fps:.1f} FPS, "
                f"{self.total_frames} frames, {self.duration:.1f}s")


class VideoWriter:
    """Wraps cv2.VideoWriter."""

    def __init__(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        self.path = path

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class KeyFrameSampler:
    """Save frames at specific progress milestones (0%, 25%, 50%, 75%, 100%)."""

    def __init__(self, output_dir, total_frames):
        self.frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.total = total_frames
        self.milestones = {0.0, 0.25, 0.5, 0.75, 1.0}
        self.saved = set()
        self.count = 0

    def maybe_save(self, frame_idx, frame):
        """Save frame if it's at a milestone."""
        if self.total <= 0:
            return
        progress = frame_idx / self.total
        for m in self.milestones:
            if m not in self.saved and abs(progress - m) < (1.0 / self.total + 0.001):
                path = os.path.join(self.frames_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(path, frame)
                self.saved.add(m)
                self.count += 1
                break
