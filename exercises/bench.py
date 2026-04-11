"""Bench press form analyzer."""

import cv2
import numpy as np
from core.angles import calculate_angle, spine_inclination, vertical_offset, symmetry_ratio, auto_angle, auto_spine_inclination
from core.reps import detect_reps
from exercises.base import ExerciseAnalyzer


class BenchAnalyzer(ExerciseAnalyzer):
    exercise_name = "bench"
    exercise_name_cn = "卧推"

    def analyze_frame(self, points, both_sides, width, height):
        shoulder = points["shoulder"]
        elbow = points["elbow"]
        wrist = points["wrist"]
        hip = points["hip"]

        # Elbow angle: shoulder-elbow-wrist
        elbow_angle = auto_angle(shoulder, elbow, wrist)

        # Shoulder angle: elbow-shoulder-hip (arm flare)
        shoulder_angle = auto_angle(elbow, shoulder, hip)

        # Back arch: spine inclination (how much the torso lifts off bench)
        back_arch = auto_spine_inclination(shoulder, hip)

        # Detect 3D mode
        is_3d = len(shoulder) >= 3 and shoulder[2] != 0.0

        # Symmetry: compare left/right elbow heights and wrist heights
        left = both_sides["left"]
        right = both_sides["right"]
        elbow_sym = symmetry_ratio(left["elbow"][1], right["elbow"][1])
        wrist_sym = symmetry_ratio(left["wrist"][1], right["wrist"][1])

        return {
            "elbow_angle": elbow_angle,
            "shoulder_angle": shoulder_angle,
            "back_arch": back_arch,
            "elbow_symmetry": elbow_sym,
            "wrist_symmetry": wrist_sym,
            "_is_3d": is_3d,
        }

    def draw_overlay(self, frame, points, angle_data, width, height):
        is_3d = angle_data.get("_is_3d", False)

        if not is_3d:
            shoulder = points["shoulder"]
            elbow = points["elbow"]
            wrist = points["wrist"]

            for pt, color in [(shoulder, (255, 0, 0)), (elbow, (0, 255, 0)),
                              (wrist, (0, 0, 255))]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, color, -1)

            cv2.putText(frame, f"Elbow: {angle_data['elbow_angle']:.0f}",
                        (int(elbow[0]) + 10, int(elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Shoulder: {angle_data['shoulder_angle']:.0f}",
                        (int(shoulder[0]) + 10, int(shoulder[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Elbow angle status (works for both 2D and 3D)
        ea = angle_data["elbow_angle"]
        if ea < 90:
            color, text = (0, 255, 0), "Good depth - full ROM"
        elif ea < 120:
            color, text = (0, 165, 255), "Moderate ROM"
        else:
            color, text = (0, 0, 255), "Limited ROM"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if is_3d:
            y_pos = 80
            for label, key, clr in [("Elbow", "elbow_angle", (0, 255, 0)),
                                     ("Shoulder", "shoulder_angle", (255, 0, 0))]:
                cv2.putText(frame, f"{label}: {angle_data[key]:.0f} deg",
                            (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
                y_pos += 35

    def generate_report(self, all_angle_data, frame_data, fps, duration):
        elbow = all_angle_data["elbow_angle"]
        shoulder = all_angle_data["shoulder_angle"]
        back = all_angle_data["back_arch"]
        e_sym = all_angle_data.get("elbow_symmetry", [1.0])
        w_sym = all_angle_data.get("wrist_symmetry", [1.0])

        is_3d = any(fd.get("backend") == "wham" for fd in frame_data) if frame_data else False

        min_elbow_idx = int(np.argmin(elbow))
        max_elbow_idx = int(np.argmax(elbow))

        lines = []
        lines.append("=" * 60)
        lines.append("     BENCH PRESS FORM ANALYSIS / 卧推动作分析报告")
        lines.append("=" * 60)
        lines.append(f"Duration: {duration:.1f}s | Frames: {len(elbow)}")
        lines.append("")

        lines.append("--- ANGLE RANGES / 角度范围 ---")
        lines.append(f"  Elbow    (肘关节):   {min(elbow):.0f} ~ {max(elbow):.0f} deg")
        lines.append(f"  Shoulder (肩关节):   {min(shoulder):.0f} ~ {max(shoulder):.0f} deg")
        lines.append(f"  Back     (背弓):     {min(back):.0f} ~ {max(back):.0f} deg (avg {np.mean(back):.0f})")
        lines.append("")

        lines.append("--- KEY POSITIONS / 关键位置 ---")
        if min_elbow_idx < len(frame_data):
            fd = frame_data[min_elbow_idx]
            lines.append(f"  Bottom (触胸): frame {fd['frame']}, t={fd['time']}s")
            lines.append(f"    Elbow={fd['elbow_angle']}, Shoulder={fd['shoulder_angle']}")
        if max_elbow_idx < len(frame_data):
            fd = frame_data[max_elbow_idx]
            lines.append(f"  Lockout (锁定): frame {fd['frame']}, t={fd['time']}s")
            lines.append(f"    Elbow={fd['elbow_angle']}, Shoulder={fd['shoulder_angle']}")
        lines.append("")

        lines.append("--- ASSESSMENT / 动作评估 ---")
        good, issues = [], []

        # Elbow ROM
        min_elbow = min(elbow)
        max_elbow = max(elbow)
        if min_elbow <= 90:
            good.append(f"  [OK] 充分触胸 (min elbow {min_elbow:.0f} deg)")
        else:
            issues.append(f"  [~] ROM不足 (min elbow {min_elbow:.0f} deg)，杠铃可能未触胸")

        # Lockout
        if max_elbow >= 160:
            good.append(f"  [OK] 完全锁定 (max elbow {max_elbow:.0f} deg)")
        else:
            issues.append(f"  [~] 锁定不完全 (max elbow {max_elbow:.0f} deg)")

        # Shoulder angle / arm flare
        avg_shoulder = np.mean(shoulder)
        if avg_shoulder > 80:
            issues.append(f"  [!] 手肘外展过大 (avg shoulder {avg_shoulder:.0f} deg)")
            issues.append("      建议：收肘至45-75度，减少肩关节压力")
        elif avg_shoulder < 30:
            issues.append(f"  [~] 手肘过度内收 (avg shoulder {avg_shoulder:.0f} deg)")
        else:
            good.append(f"  [OK] 手肘角度合理 (avg shoulder {avg_shoulder:.0f} deg)")

        # Symmetry
        avg_e_sym = np.mean(e_sym)
        avg_w_sym = np.mean(w_sym)
        if avg_e_sym < 0.85 or avg_w_sym < 0.85:
            issues.append(f"  [~] 左右不对称 (elbow sym={avg_e_sym:.2f}, wrist sym={avg_w_sym:.2f})")
            issues.append("      建议：检查握距是否居中，注意双侧均匀发力")
        else:
            good.append(f"  [OK] 左右对称性良好 (sym={avg_e_sym:.2f})")

        if good:
            lines.append("  优点:")
            lines.extend(good)
        if issues:
            lines.append("  问题与建议:")
            lines.extend(issues)
        if not issues:
            lines.append("  整体动作质量不错!")

        lines.append("")
        lines.append("=" * 60)
        if is_3d:
            lines.append("基于WHAM 3D姿态估计，角度不受拍摄角度影响。")
        else:
            lines.append("注意：基于2D姿态估计，结果受拍摄角度影响。")
            lines.append("建议从侧面或斜上方45度拍摄卧推。正面拍摄可检测对称性。")
        lines.append("=" * 60)
        return "\n".join(lines)

    def chart_config(self):
        return (
            ["elbow_angle", "shoulder_angle", "back_arch"],
            ["Elbow / 肘关节", "Shoulder / 肩关节", "Back arch / 背弓"],
            "Bench Press Angle Analysis / 卧推角度分析",
        )

    def get_rep_peaks(self, all_angle_data, fps):
        elbow = all_angle_data.get("elbow_angle", [])
        if not elbow:
            return None
        # For bench, count valleys (lowest elbow angle = bar at chest)
        reps_info = detect_reps(elbow, mode="valley", min_prominence=20,
                                min_distance_sec=1.0, fps=fps)
        return reps_info["peak_frames"] if reps_info["count"] > 0 else None
