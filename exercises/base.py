"""Base class for exercise analyzers."""

import os
import json
import cv2
import numpy as np

from core.pose import create_pose, pick_visible_side, extract_points, extract_both_sides, draw_pose
from core.video import VideoReader, VideoWriter, KeyFrameSampler
from core.chart import generate_angle_chart


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
        """Compute angle metrics for a single frame.

        Args:
            points: dict of name->(x,y) for the primary visible side.
            both_sides: {"left": {...}, "right": {...}} for symmetry checks.
            width, height: frame dimensions.

        Returns:
            dict with angle values (keys must be consistent across frames).
        """
        raise NotImplementedError

    def draw_overlay(self, frame, points, angle_data, width, height):
        """Draw exercise-specific angle labels and warnings on the frame."""
        raise NotImplementedError

    def generate_report(self, all_angle_data, frame_data, fps, duration):
        """Generate a text report string from collected data.

        Args:
            all_angle_data: dict of angle_name -> list of values across frames.
            frame_data: list of per-frame dicts (frame, time, angles, side).
            fps: video FPS.
            duration: video duration in seconds.

        Returns:
            Report string.
        """
        raise NotImplementedError

    def chart_config(self):
        """Return (angle_keys, labels, title) for chart generation."""
        raise NotImplementedError

    def get_rep_peaks(self, all_angle_data, fps):
        """Return list of frame indices at rep peaks (for chart markers).

        Subclasses may override to enable rep detection. Default: no reps.
        """
        return None

    def run(self, video_path, output_dir, generate_video=True, generate_charts=True):
        """Execute the full analysis pipeline.

        Args:
            video_path: path to input video.
            output_dir: directory to write outputs.
            generate_video: whether to produce annotated video.
            generate_charts: whether to produce angle charts.
        """
        os.makedirs(output_dir, exist_ok=True)

        with VideoReader(video_path) as reader:
            print(reader.info_str())

            writer = None
            if generate_video:
                video_out = os.path.join(output_dir, "annotated.mp4")
                writer = VideoWriter(video_out, reader.fps, reader.width, reader.height)

            sampler = KeyFrameSampler(output_dir, reader.total_frames)
            pose = create_pose()

            frame_data = []
            all_angle_data = {}  # key -> list of values

            for frame_idx, frame in reader.frames():
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    side_name, side_lms = pick_visible_side(lms)
                    points = extract_points(lms, side_lms, reader.width, reader.height)
                    both_sides = extract_both_sides(lms, reader.width, reader.height)

                    angle_data = self.analyze_frame(points, both_sides, reader.width, reader.height)

                    # Accumulate
                    for key, val in angle_data.items():
                        all_angle_data.setdefault(key, []).append(val)

                    fd = {
                        "frame": frame_idx,
                        "time": round(frame_idx / reader.fps, 2),
                        "side": side_name,
                    }
                    fd.update({k: round(v, 1) for k, v in angle_data.items()})
                    frame_data.append(fd)

                    # Draw
                    draw_pose(frame, results.pose_landmarks)
                    self.draw_overlay(frame, points, angle_data, reader.width, reader.height)

                # Frame counter
                cv2.putText(frame, f"Frame {frame_idx}/{reader.total_frames}",
                            (20, reader.height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if writer:
                    writer.write(frame)
                sampler.maybe_save(frame_idx, frame)

            pose.close()
            if writer:
                writer.close()

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

        # Charts
        if generate_charts and frame_data:
            chart_path = os.path.join(output_dir, "charts.png")
            keys, labels, title = self.chart_config()
            rep_peaks = self.get_rep_peaks(all_angle_data, reader.fps)
            generate_angle_chart(frame_data, keys, labels, title, chart_path, reader.fps,
                                 rep_peak_frames=rep_peaks)

        # Summary
        print(f"\nDone! Output -> {output_dir}")
        if generate_video:
            print(f"  annotated.mp4")
        print(f"  report.txt")
        print(f"  frame_data.json")
        if generate_charts:
            print(f"  charts.png")
        print(f"  frames/ ({sampler.count} key frames)")
        print()
        print(report)
