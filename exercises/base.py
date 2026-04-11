"""Base class for exercise analyzers."""

import os
import json
import cv2
import numpy as np

from core.pose import create_pose, pick_visible_side, extract_points, extract_both_sides, draw_pose
from core.video import VideoReader, VideoWriter, KeyFrameSampler
from core.chart import generate_angle_chart, generate_bar_path_chart, generate_velocity_chart


class ExerciseAnalyzer:
    """Abstract base for per-exercise analysis logic.

    Subclasses must implement:
        - analyze_frame(points, both_sides, width, height) -> dict of angle data
        - draw_overlay(frame, points, angle_data, width, height)
        - generate_report(all_angle_data, frame_data, fps, duration) -> str
        - chart_config() -> (angle_keys, labels, title)
    """

    exercise_name = "exercise"
    exercise_name_cn = "运动"

    def analyze_frame(self, points, both_sides, width, height):
        raise NotImplementedError

    def draw_overlay(self, frame, points, angle_data, width, height):
        raise NotImplementedError

    def generate_report(self, all_angle_data, frame_data, fps, duration):
        raise NotImplementedError

    def chart_config(self):
        raise NotImplementedError

    def get_rep_peaks(self, all_angle_data, fps):
        return None

    def get_rep_ranges(self, all_angle_data, fps):
        """Return list of (start, peak, end) tuples for bar path / velocity charts.

        Subclasses may override. Default: derive from get_rep_peaks.
        """
        return None

    def run(self, video_path, output_dir, generate_video=True, generate_charts=True,
            backend="mediapipe", barbell_model=None):
        """Execute the full analysis pipeline.

        Args:
            video_path: path to input video.
            output_dir: directory to write outputs.
            generate_video: whether to produce annotated video.
            generate_charts: whether to produce angle charts.
            backend: "mediapipe" (2D) or "wham" (3D).
            barbell_model: path to YOLOv8 barbell model, or None to skip.
        """
        os.makedirs(output_dir, exist_ok=True)

        if backend == "wham":
            self._run_wham(video_path, output_dir, generate_video, generate_charts,
                           barbell_model)
        else:
            self._run_mediapipe(video_path, output_dir, generate_video, generate_charts,
                                barbell_model)

    def _run_mediapipe(self, video_path, output_dir, generate_video, generate_charts,
                       barbell_model):
        """MediaPipe 2D pipeline (original)."""
        # Optional barbell tracker
        barbell_tracker = None
        if barbell_model:
            try:
                from core.barbell import BarbellTracker
                barbell_tracker = BarbellTracker(model_path=barbell_model)
                print("  [barbell] YOLOv8 barbell tracking enabled")
            except Exception as e:
                print(f"  [barbell] Failed to load: {e}. Using wrist fallback.")

        with VideoReader(video_path) as reader:
            print(reader.info_str())

            writer = None
            if generate_video:
                video_out = os.path.join(output_dir, "annotated.mp4")
                writer = VideoWriter(video_out, reader.fps, reader.width, reader.height)

            sampler = KeyFrameSampler(output_dir, reader.total_frames)
            pose = create_pose()

            frame_data = []
            all_angle_data = {}

            for frame_idx, frame in reader.frames():
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results is not None:
                    lms = results.landmark
                    side_name, side_lms = pick_visible_side(lms)
                    points = extract_points(lms, side_lms, reader.width, reader.height)
                    both_sides = extract_both_sides(lms, reader.width, reader.height)

                    # Barbell detection (override wrist-based bar position)
                    if barbell_tracker:
                        wrist_pos = (points["wrist"][0], points["wrist"][1])
                        bar_det = barbell_tracker.detect(frame, wrist_fallback=wrist_pos)
                        # Inject barbell position into points for analyzers
                        points["_bar_detected"] = bar_det

                    angle_data = self.analyze_frame(points, both_sides,
                                                    reader.width, reader.height)

                    for key, val in angle_data.items():
                        all_angle_data.setdefault(key, []).append(val)

                    fd = {
                        "frame": frame_idx,
                        "time": round(frame_idx / reader.fps, 2),
                        "side": side_name,
                    }
                    fd.update({k: round(v, 1) for k, v in angle_data.items()
                               if isinstance(v, (int, float))})
                    frame_data.append(fd)

                    draw_pose(frame, results)
                    self.draw_overlay(frame, points, angle_data, reader.width, reader.height)

                    # Draw barbell bbox if detected
                    if barbell_tracker and bar_det.get("bbox"):
                        x1, y1, x2, y2 = bar_det["bbox"]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                      (0, 255, 255), 2)
                        cv2.putText(frame,
                                    f"BAR {bar_det['confidence']:.0%}",
                                    (int(x1), int(y1) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.putText(frame, f"Frame {frame_idx}/{reader.total_frames}",
                            (20, reader.height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if writer:
                    writer.write(frame)
                sampler.maybe_save(frame_idx, frame)

            pose.close()
            if writer:
                writer.close()

        if barbell_tracker:
            stats = barbell_tracker.get_stats()
            print(f"  [barbell] Detection rate: {stats['detection_rate']:.0%} "
                  f"({stats['yolo_detections']}/{stats['total_frames']} frames)")

        self._finalize(frame_data, all_angle_data, output_dir, reader, generate_charts,
                       backend="mediapipe")

    def _run_wham(self, video_path, output_dir, generate_video, generate_charts,
                  barbell_model):
        """WHAM 3D pipeline."""
        from core.pose_3d import (WHAMPoseExtractor, extract_points_3d,
                                  extract_both_sides_3d, pick_visible_side_3d)

        print("  Backend: WHAM 3D")
        extractor = WHAMPoseExtractor()
        wham_result = extractor.process(video_path)
        joints_3d = wham_result["joints_3d"]
        frame_ids = wham_result["frame_ids"]

        # Build frame_id → wham_idx lookup for sparse frame mapping
        frame_id_set = set(int(f) for f in frame_ids)

        with VideoReader(video_path) as reader:
            print(reader.info_str())

            writer = None
            if generate_video:
                video_out = os.path.join(output_dir, "annotated.mp4")
                writer = VideoWriter(video_out, reader.fps, reader.width, reader.height)

            sampler = KeyFrameSampler(output_dir, reader.total_frames)

            frame_data = []
            all_angle_data = {}
            wham_idx = 0

            for frame_idx, frame in reader.frames():
                # Match video frame to WHAM output via frame_ids
                if wham_idx < len(frame_ids) and frame_idx == int(frame_ids[wham_idx]):
                    j3d = joints_3d[wham_idx]
                    side_name, _ = pick_visible_side_3d(j3d)
                    points = extract_points_3d(j3d, side_name)
                    both_sides = extract_both_sides_3d(j3d)

                    angle_data = self.analyze_frame(points, both_sides,
                                                    reader.width, reader.height)

                    for key, val in angle_data.items():
                        all_angle_data.setdefault(key, []).append(val)

                    fd = {
                        "frame": frame_idx,
                        "time": round(frame_idx / reader.fps, 2),
                        "side": side_name,
                        "backend": "wham",
                    }
                    fd.update({k: round(v, 1) for k, v in angle_data.items()
                               if isinstance(v, (int, float))})
                    frame_data.append(fd)

                    self.draw_overlay(frame, points, angle_data, reader.width, reader.height)
                    wham_idx += 1

                cv2.putText(frame, f"Frame {frame_idx}/{reader.total_frames} [WHAM 3D]",
                            (20, reader.height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if writer:
                    writer.write(frame)
                sampler.maybe_save(frame_idx, frame)

            if writer:
                writer.close()

        self._finalize(frame_data, all_angle_data, output_dir, reader, generate_charts,
                       backend="wham")

    def _finalize(self, frame_data, all_angle_data, output_dir, reader, generate_charts,
                  backend="mediapipe"):
        """Write report, JSON, and charts."""
        if not frame_data:
            print("ERROR: No pose detected in video!")
            return

        # Report
        report = self.generate_report(all_angle_data, frame_data, reader.fps, reader.duration)
        report_path = os.path.join(output_dir, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Frame data JSON
        data_path = os.path.join(output_dir, "frame_data.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)

        # Angle charts
        if generate_charts and frame_data:
            chart_path = os.path.join(output_dir, "charts.png")
            keys, labels, title = self.chart_config()
            rep_peaks = self.get_rep_peaks(all_angle_data, reader.fps)
            generate_angle_chart(frame_data, keys, labels, title, chart_path, reader.fps,
                                 rep_peak_frames=rep_peaks)

            # Bar path chart (if bar tracking data exists)
            if "bar_x_px" in all_angle_data and "bar_y_px" in all_angle_data:
                bar_path_out = os.path.join(output_dir, "bar_path.png")
                rep_ranges = self.get_rep_ranges(all_angle_data, reader.fps) or []
                generate_bar_path_chart(frame_data, rep_ranges, bar_path_out,
                                        reader.width, reader.height)

                # Velocity chart
                velocity_out = os.path.join(output_dir, "velocity.png")
                generate_velocity_chart(frame_data, rep_ranges, velocity_out, reader.fps)

        # Summary
        print(f"\nDone! Output -> {output_dir}")
        outputs = ["report.txt", "frame_data.json"]
        if generate_charts:
            outputs.append("charts.png")
            if "bar_x_px" in all_angle_data:
                outputs.extend(["bar_path.png", "velocity.png"])
        for o in outputs:
            print(f"  {o}")
        print()
        print(report)
