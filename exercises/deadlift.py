"""Deadlift form analyzer with barbell tracking via wrist position."""

import cv2
import numpy as np
from core.angles import calculate_angle, spine_inclination
from core.reps import detect_reps
from exercises.base import ExerciseAnalyzer


class DeadliftAnalyzer(ExerciseAnalyzer):
    exercise_name = "deadlift"
    exercise_name_cn = "硬拉"

    def analyze_frame(self, points, both_sides, width, height):
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        ankle = points["ankle"]
        wrist = points["wrist"]

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        back_angle = spine_inclination(shoulder, hip)
        bar_drift = abs(shoulder[0] - ankle[0]) / width * 100

        # Barbell tracking: use wrist Y position (normalized, 0=top 100=bottom)
        # In image coords, y increases downward, so higher bar = smaller y
        bar_y = wrist[1] / height * 100  # 0% = top of frame, 100% = bottom

        # Also track both wrists for bar path visualization
        left_wrist = both_sides["left"]["wrist"]
        right_wrist = both_sides["right"]["wrist"]
        # Average of both wrists as bar center
        bar_center_x = (left_wrist[0] + right_wrist[0]) / 2
        bar_center_y = (left_wrist[1] + right_wrist[1]) / 2

        return {
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "back_angle": back_angle,
            "bar_drift": bar_drift,
            "bar_y": bar_y,
            "bar_x_px": bar_center_x,
            "bar_y_px": bar_center_y,
        }

    def draw_overlay(self, frame, points, angle_data, width, height):
        shoulder = points["shoulder"]
        hip = points["hip"]
        knee = points["knee"]
        wrist = points["wrist"]

        for pt, color in [(shoulder, (255, 0, 0)), (hip, (0, 255, 0)),
                          (knee, (0, 0, 255)), (points["ankle"], (255, 255, 0))]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 8, color, -1)

        # Highlight wrist/bar position with a larger marker
        cv2.circle(frame, (int(wrist[0]), int(wrist[1])), 10, (0, 255, 255), 3)
        cv2.putText(frame, "BAR", (int(wrist[0]) + 12, int(wrist[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(frame, f"Knee: {angle_data['knee_angle']:.0f}",
                    (int(knee[0]) + 10, int(knee[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Hip: {angle_data['hip_angle']:.0f}",
                    (int(hip[0]) + 10, int(hip[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Back: {angle_data['back_angle']:.0f}",
                    (int(shoulder[0]) + 10, int(shoulder[1]) - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        ba = angle_data["back_angle"]
        if ba > 45:
            color, text = (0, 0, 255), "WARNING: Excessive forward lean!"
        elif ba > 30:
            color, text = (0, 165, 255), "Moderate forward lean"
        else:
            color, text = (0, 255, 0), "Good back position"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def generate_report(self, all_angle_data, frame_data, fps, duration):
        knee = all_angle_data["knee_angle"]
        hip = all_angle_data["hip_angle"]
        back = all_angle_data["back_angle"]
        bar_y = all_angle_data["bar_y"]

        # --- Rep detection using barbell Y position ---
        # bar_y: 0=top, 100=bottom. A lockout = bar at its highest = bar_y valley
        # So we detect valleys in bar_y to find lockout moments
        reps_info = detect_reps(bar_y, mode="valley", min_prominence=8,
                                min_distance_sec=2.0, fps=fps)

        # Filter out incomplete reps: bar must move at least 15% of frame height
        valid_reps = []
        for start, peak, end in reps_info["rep_ranges"]:
            rep_bar = bar_y[start:end + 1]
            if not rep_bar:
                continue
            bar_travel = max(rep_bar) - min(rep_bar)
            if bar_travel >= 15:  # at least 15% of frame height movement
                valid_reps.append((start, peak, end))

        rep_count = len(valid_reps)

        min_hip_idx = int(np.argmin(hip))
        max_hip_idx = int(np.argmax(hip))

        lines = []
        lines.append("=" * 60)
        lines.append("     DEADLIFT FORM ANALYSIS / 硬拉动作分析报告")
        lines.append("=" * 60)
        lines.append(f"Duration: {duration:.1f}s | Frames: {len(knee)}")
        lines.append(f"Reps detected / 检测到次数: {rep_count}")
        lines.append(f"  (via barbell tracking / 通过杠铃轨迹检测)")
        lines.append("")

        # --- Per-rep breakdown ---
        if rep_count > 0:
            lines.append("--- PER-REP BREAKDOWN / 逐次分析 ---")
            lines.append(f"  {'Rep':>3}  {'Time':>8}  {'Bar Travel':>10}  "
                         f"{'Hip ROM':>8}  {'Max Back':>9}  {'Lockout':>8}  {'Grade':>6}")
            lines.append(f"  {'---':>3}  {'----':>8}  {'----------':>10}  "
                         f"{'-------':>8}  {'---------':>9}  {'-------':>8}  {'-----':>6}")

            for i, (start, peak, end) in enumerate(valid_reps):
                rep_hip = hip[start:end + 1]
                rep_back = back[start:end + 1]
                rep_bar = bar_y[start:end + 1]

                if not rep_hip:
                    continue

                r_bar_travel = max(rep_bar) - min(rep_bar)
                r_hip_rom = max(rep_hip) - min(rep_hip)
                r_max_back = max(rep_back)
                r_lockout_hip = max(rep_hip)
                r_time = frame_data[peak]["time"] if peak < len(frame_data) else 0

                grade = self._grade_rep(r_hip_rom, r_max_back, r_lockout_hip)

                lines.append(f"  #{i+1:>2}  {r_time:>6.1f}s  {r_bar_travel:>8.1f}%"
                             f"  {r_hip_rom:>6.0f} deg  {r_max_back:>7.0f} deg"
                             f"  {r_lockout_hip:>6.0f} deg  {grade:>6}")

            lines.append("")

        # --- Bar path summary ---
        bar_y_px = all_angle_data["bar_y_px"]
        bar_x_px = all_angle_data["bar_x_px"]
        lines.append("--- BAR PATH / 杠铃轨迹 ---")
        lines.append(f"  Bar Y range: {min(bar_y):.1f}% ~ {max(bar_y):.1f}% of frame")
        lines.append(f"  Bar X drift: {max(bar_x_px) - min(bar_x_px):.0f} px horizontal")
        lines.append("")

        # --- Overall angle ranges ---
        lines.append("--- ANGLE RANGES (OVERALL) / 总体角度范围 ---")
        lines.append(f"  Knee  (膝关节):   {min(knee):.0f} ~ {max(knee):.0f} deg")
        lines.append(f"  Hip   (髋关节):   {min(hip):.0f} ~ {max(hip):.0f} deg")
        lines.append(f"  Back  (脊柱前倾): {min(back):.0f} ~ {max(back):.0f} deg (avg {np.mean(back):.0f})")
        lines.append("")

        # --- Key positions ---
        lines.append("--- KEY POSITIONS / 关键位置 ---")
        if min_hip_idx < len(frame_data):
            fd = frame_data[min_hip_idx]
            lines.append(f"  Bottom (起始): frame {fd['frame']}, t={fd['time']}s")
            lines.append(f"    Hip={fd['hip_angle']}, Knee={fd['knee_angle']}, Back={fd['back_angle']}")
        if max_hip_idx < len(frame_data):
            fd = frame_data[max_hip_idx]
            lines.append(f"  Lockout (锁定): frame {fd['frame']}, t={fd['time']}s")
            lines.append(f"    Hip={fd['hip_angle']}, Knee={fd['knee_angle']}, Back={fd['back_angle']}")
        lines.append("")

        # --- Overall assessment ---
        lines.append("--- ASSESSMENT / 动作评估 ---")
        good, issues = [], []

        max_back = max(back)
        if max_back > 55:
            issues.append(f"  [!] 脊柱前倾过大 (max {max_back:.0f} deg)，注意腰椎风险")
            issues.append("      建议：加强核心稳定性，挺胸收紧背部")
        elif max_back > 40:
            issues.append(f"  [~] 脊柱前倾适中 ({max_back:.0f} deg)，注意保持中立位")
        else:
            good.append(f"  [OK] 脊柱前倾控制良好 (max {max_back:.0f} deg)")

        hip_rom = max(hip) - min(hip)
        if hip_rom < 40:
            issues.append(f"  [~] 髋关节ROM较小 ({hip_rom:.0f} deg)，髋铰链可能不充分")
        else:
            good.append(f"  [OK] 髋关节ROM充分 ({hip_rom:.0f} deg)")

        if max(hip) > 165:
            good.append(f"  [OK] 完全锁定 (hip {max(hip):.0f} deg)")
        else:
            issues.append(f"  [~] 锁定不完全 (max hip {max(hip):.0f} deg)")

        # Fatigue detection
        if rep_count >= 3:
            first_range = valid_reps[0]
            last_range = valid_reps[-1]
            first_back = max(back[first_range[0]:first_range[2] + 1])
            last_back = max(back[last_range[0]:last_range[2] + 1])
            if last_back - first_back > 10:
                issues.append(f"  [~] 疲劳迹象: 最后一个rep脊柱前倾({last_back:.0f} deg)"
                              f"比第一个({first_back:.0f} deg)增大{last_back - first_back:.0f} deg")
                issues.append("      后几组注意控制动作质量，必要时降低重量")

        # Bar path straightness
        bar_x_range = max(bar_x_px) - min(bar_x_px)
        if bar_x_range < 50:
            good.append(f"  [OK] 杠铃轨迹较直 (水平漂移 {bar_x_range:.0f} px)")
        else:
            issues.append(f"  [~] 杠铃轨迹水平漂移较大 ({bar_x_range:.0f} px)")
            issues.append("      建议：注意杠铃贴身运动，避免前后漂移")

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
        lines.append("注意：基于2D姿态估计+手腕追踪，结果受拍摄角度影响。")
        lines.append("建议侧面拍摄以获得最佳分析效果。")
        lines.append("=" * 60)
        return "\n".join(lines)

    def _grade_rep(self, hip_rom, max_back, lockout_hip):
        """Grade a single rep: A/B/C."""
        score = 0
        if hip_rom >= 60:
            score += 1
        if max_back <= 50:
            score += 1
        if lockout_hip >= 160:
            score += 1

        if score == 3:
            return "A"
        elif score == 2:
            return "B"
        else:
            return "C"

    def chart_config(self):
        return (
            ["knee_angle", "hip_angle", "back_angle", "bar_y"],
            ["Knee / 膝关节", "Hip / 髋关节", "Back / 脊柱前倾", "Bar height / 杠铃高度(%)"],
            "Deadlift Angle Analysis / 硬拉角度分析",
        )

    def get_rep_peaks(self, all_angle_data, fps):
        bar_y = all_angle_data.get("bar_y", [])
        if not bar_y:
            return None
        reps_info = detect_reps(bar_y, mode="valley", min_prominence=8,
                                min_distance_sec=2.0, fps=fps)
        # Filter by bar travel
        valid_peaks = []
        for start, peak, end in reps_info["rep_ranges"]:
            rep_bar = bar_y[start:end + 1]
            if rep_bar and (max(rep_bar) - min(rep_bar)) >= 15:
                valid_peaks.append(peak)
        return valid_peaks if valid_peaks else None
