"""Squat form analyzer."""

import cv2
import numpy as np
from core.angles import calculate_angle, spine_inclination, horizontal_offset
from core.reps import detect_reps
from exercises.base import ExerciseAnalyzer


class SquatAnalyzer(ExerciseAnalyzer):
    exercise_name = "squat"
    exercise_name_cn = "深蹲"

    def analyze_frame(self, points, both_sides, width, height):
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        ankle = points["ankle"]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        back_angle = spine_inclination(shoulder, hip)

        # Knee valgus: compare knee-ankle horizontal offset (both sides)
        left = both_sides["left"]
        right = both_sides["right"]
        l_valgus = left["knee"][0] - left["ankle"][0]  # positive = knee inside
        r_valgus = right["ankle"][0] - right["knee"][0]
        knee_valgus = (l_valgus + r_valgus) / 2  # avg inward drift in pixels

        # Knee over toe: horizontal distance between knee and foot_index
        knee_over_toe = knee[0] - points.get("foot_index", ankle)[0]

        return {
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "back_angle": back_angle,
            "knee_valgus_px": knee_valgus,
        }

    def draw_overlay(self, frame, points, angle_data, width, height):
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        ankle = points["ankle"]

        for pt, color in [(shoulder, (255, 0, 0)), (hip, (0, 255, 0)),
                          (knee, (0, 0, 255)), (ankle, (255, 255, 0))]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, color, -1)

        cv2.putText(frame, f"Knee: {angle_data['knee_angle']:.0f}",
                    (int(knee[0]) + 10, int(knee[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Hip: {angle_data['hip_angle']:.0f}",
                    (int(hip[0]) + 10, int(hip[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Back: {angle_data['back_angle']:.0f}",
                    (int(shoulder[0]) + 10, int(shoulder[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Depth indicator
        ka = angle_data["knee_angle"]
        if ka <= 90:
            color, text = (0, 255, 0), "Good depth!"
        elif ka <= 110:
            color, text = (0, 165, 255), "Parallel - could go deeper"
        else:
            color, text = (0, 0, 255), "Above parallel"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def generate_report(self, all_angle_data, frame_data, fps, duration):
        knee = all_angle_data["knee_angle"]
        hip = all_angle_data["hip_angle"]
        back = all_angle_data["back_angle"]
        valgus = all_angle_data.get("knee_valgus_px", [0])

        min_knee_idx = int(np.argmin(knee))

        lines = []
        lines.append("=" * 60)
        lines.append("     SQUAT FORM ANALYSIS / 深蹲动作分析报告")
        lines.append("=" * 60)
        lines.append(f"Duration: {duration:.1f}s | Frames: {len(knee)}")
        lines.append("")

        lines.append("--- ANGLE RANGES / 角度范围 ---")
        lines.append(f"  Knee  (膝关节):   {min(knee):.0f} ~ {max(knee):.0f} deg")
        lines.append(f"  Hip   (髋关节):   {min(hip):.0f} ~ {max(hip):.0f} deg")
        lines.append(f"  Back  (脊柱前倾): {min(back):.0f} ~ {max(back):.0f} deg (avg {np.mean(back):.0f})")
        lines.append("")

        lines.append("--- KEY POSITIONS / 关键位置 ---")
        if min_knee_idx < len(frame_data):
            fd = frame_data[min_knee_idx]
            lines.append(f"  Hole (最低点): frame {fd['frame']}, t={fd['time']}s")
            lines.append(f"    Knee={fd['knee_angle']}, Hip={fd['hip_angle']}, Back={fd['back_angle']}")
        lines.append("")

        lines.append("--- ASSESSMENT / 动作评估 ---")
        good, issues = [], []

        # Depth
        min_knee = min(knee)
        if min_knee <= 90:
            good.append(f"  [OK] 蹲到位 (min knee {min_knee:.0f} deg, below parallel)")
        elif min_knee <= 110:
            issues.append(f"  [~] 勉强到平行 (min knee {min_knee:.0f} deg)")
            issues.append("      建议：增加踝关节灵活性，尝试稍宽站距")
        else:
            issues.append(f"  [!] 深度不足 (min knee {min_knee:.0f} deg)")
            issues.append("      建议：降低重量，练习徒手全蹲，加强踝/髋灵活性")

        # Knee valgus
        max_valgus = max(valgus) if valgus else 0
        if max_valgus > 30:
            issues.append(f"  [!] 膝盖内扣明显 (max {max_valgus:.0f} px)")
            issues.append("      建议：加强臀中肌，练习弹力带深蹲")
        elif max_valgus > 15:
            issues.append(f"  [~] 轻微膝盖内扣 ({max_valgus:.0f} px)")
        else:
            good.append("  [OK] 膝盖跟踪良好，无明显内扣")

        # Back angle (Good morning squat check)
        max_back = max(back)
        if max_back > 60:
            issues.append(f"  [!] 过度前倾 (max {max_back:.0f} deg)，可能变成 'Good morning squat'")
            issues.append("      建议：加强核心和竖脊肌，注意挺胸")
        elif max_back > 45:
            issues.append(f"  [~] 前倾偏大 ({max_back:.0f} deg)，注意保持上半身相对直立")
        else:
            good.append(f"  [OK] 躯干控制良好 (max back {max_back:.0f} deg)")

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
        lines.append("注意：基于2D姿态估计，结果受拍摄角度影响。建议侧面拍摄。")
        lines.append("正面拍摄可更好检测膝盖内扣。")
        lines.append("=" * 60)
        return "\n".join(lines)

    def chart_config(self):
        return (
            ["knee_angle", "hip_angle", "back_angle"],
            ["Knee / 膝关节", "Hip / 髋关节", "Back / 脊柱前倾"],
            "Squat Angle Analysis / 深蹲角度分析",
        )

    def get_rep_peaks(self, all_angle_data, fps):
        knee = all_angle_data.get("knee_angle", [])
        if not knee:
            return None
        # For squats, count valleys (lowest knee angle = bottom of squat)
        reps_info = detect_reps(knee, mode="valley", min_prominence=25,
                                min_distance_sec=1.5, fps=fps)
        return reps_info["peak_frames"] if reps_info["count"] > 0 else None
