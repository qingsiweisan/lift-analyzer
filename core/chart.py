"""Chart generation for angle data over time."""

import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

# Find a Chinese-capable font from system fonts
_CN_FONT = None
_CN_CANDIDATES = {'microsoft yahei', 'simhei', 'simsun', 'dengxian', 'noto sans sc'}
for f in fontManager.ttflist:
    if f.name.lower() in _CN_CANDIDATES:
        _CN_FONT = f.name
        break

if _CN_FONT:
    plt.rcParams['font.sans-serif'] = [_CN_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def generate_angle_chart(frame_data, angle_keys, labels, title, output_path, fps,
                         rep_peak_frames=None):
    """Generate a multi-line angle-vs-time chart with optional rep markers.

    Args:
        frame_data: list of dicts, each with 'frame' key and angle keys.
        angle_keys: list of dict keys to plot (e.g. ['knee_angle', 'hip_angle']).
        labels: list of display labels corresponding to angle_keys.
        title: chart title string.
        output_path: path to save the PNG.
        fps: video FPS for time axis conversion.
        rep_peak_frames: optional list of frame indices where reps peak (for markers).
    """
    if not frame_data:
        return

    times = [d["frame"] / fps for d in frame_data]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (key, label) in enumerate(zip(angle_keys, labels)):
        values = [d[key] for d in frame_data]
        color = colors[i % len(colors)]
        ax.plot(times, values, color=color, linewidth=1.2, label=label, alpha=0.85)

    # Mark rep peaks on the chart
    if rep_peak_frames:
        for rep_num, peak_idx in enumerate(rep_peak_frames, 1):
            if peak_idx < len(times):
                t = times[peak_idx]
                ax.axvline(x=t, color='#7f8c8d', linestyle='--', linewidth=0.8, alpha=0.5)
                ax.text(t, 195, f'Rep {rep_num}', ha='center', va='top',
                        fontsize=8, color='#2c3e50', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1',
                                  edgecolor='#7f8c8d', alpha=0.8))

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Angle (degrees)", fontsize=11)

    rep_suffix = f" ({len(rep_peak_frames)} reps)" if rep_peak_frames else ""
    ax.set_title(title + rep_suffix, fontsize=13, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 200)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_bar_path_chart(frame_data, rep_ranges, output_path, width, height):
    """Generate a bar path (X-Y trajectory) chart with per-rep overlay.

    Args:
        frame_data: list of dicts with 'bar_x_px' and 'bar_y_px' keys.
        rep_ranges: list of (start, peak, end) tuples from rep detection.
        output_path: path to save the PNG.
        width, height: video frame dimensions (for normalization).
    """
    if not frame_data or not any("bar_x_px" in d for d in frame_data):
        return

    bar_x = [d.get("bar_x_px", 0) for d in frame_data]
    bar_y = [d.get("bar_y_px", 0) for d in frame_data]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: full trajectory
    ax1 = axes[0]
    ax1.plot(bar_x, bar_y, color='#3498db', linewidth=0.8, alpha=0.5, label='Full path')
    ax1.scatter([bar_x[0]], [bar_y[0]], color='#2ecc71', s=80, zorder=5, label='Start')
    ax1.scatter([bar_x[-1]], [bar_y[-1]], color='#e74c3c', s=80, zorder=5, label='End')
    ax1.set_xlabel("X (px)")
    ax1.set_ylabel("Y (px)")
    ax1.set_title("Bar Path (Full) / 杠铃轨迹（全程）")
    ax1.invert_yaxis()  # image coords: y increases downward
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # Right: per-rep overlay
    ax2 = axes[1]
    rep_colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e']

    if rep_ranges:
        for i, (start, peak, end) in enumerate(rep_ranges):
            s = max(0, start)
            e = min(len(bar_x) - 1, end)
            rx = bar_x[s:e + 1]
            ry = bar_y[s:e + 1]
            if not rx:
                continue
            # Normalize: shift so start Y is at 0
            ry_norm = [y - ry[0] for y in ry]
            rx_norm = [x - rx[0] for x in rx]
            color = rep_colors[i % len(rep_colors)]
            ax2.plot(rx_norm, ry_norm, color=color, linewidth=1.5, label=f'Rep {i+1}')
    else:
        ax2.text(0.5, 0.5, 'No reps detected', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12, color='#7f8c8d')

    ax2.set_xlabel("X offset (px)")
    ax2.set_ylabel("Y offset (px)")
    ax2.set_title("Bar Path (Per Rep) / 杠铃轨迹（逐次叠加）")
    ax2.invert_yaxis()
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_velocity_chart(frame_data, rep_ranges, output_path, fps):
    """Generate velocity and acceleration curves from bar Y position.

    Args:
        frame_data: list of dicts with 'bar_y_px' key.
        rep_ranges: list of (start, peak, end) tuples.
        output_path: path to save the PNG.
        fps: video FPS for time derivative.
    """
    if not frame_data or not any("bar_y_px" in d for d in frame_data):
        return

    import numpy as np

    bar_y = np.array([d.get("bar_y_px", 0) for d in frame_data], dtype=float)
    times = np.array([d["frame"] / fps for d in frame_data])

    # Smooth bar_y before differentiation
    kernel = max(3, min(15, len(bar_y) // 20))
    if kernel % 2 == 0:
        kernel += 1
    bar_y_smooth = np.convolve(bar_y, np.ones(kernel) / kernel, mode='same')

    # Velocity: dy/dt (pixels/sec). Negative = bar moving up (concentric)
    dt = 1.0 / fps
    velocity = np.gradient(bar_y_smooth, dt)
    # Invert so positive = upward (concentric)
    velocity = -velocity

    # Acceleration: dv/dt
    acceleration = np.gradient(velocity, dt)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Velocity
    ax1.plot(times, velocity, color='#3498db', linewidth=1.0, alpha=0.8)
    ax1.axhline(y=0, color='#7f8c8d', linewidth=0.5, linestyle='-')
    ax1.fill_between(times, velocity, 0, where=(velocity > 0),
                     color='#2ecc71', alpha=0.2, label='Concentric (up)')
    ax1.fill_between(times, velocity, 0, where=(velocity < 0),
                     color='#e74c3c', alpha=0.2, label='Eccentric (down)')

    # Mark rep peaks
    if rep_ranges:
        for i, (start, peak, end) in enumerate(rep_ranges):
            if peak < len(times):
                ax1.axvline(x=times[peak], color='#7f8c8d', linestyle='--',
                            linewidth=0.8, alpha=0.5)
                ax1.text(times[peak], ax1.get_ylim()[1] * 0.9, f'R{i+1}',
                         ha='center', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='#ecf0f1',
                                   edgecolor='#7f8c8d', alpha=0.8))

    ax1.set_ylabel("Velocity (px/s)")
    ax1.set_title("Bar Velocity / 杠铃速度", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Acceleration
    ax2.plot(times, acceleration, color='#f39c12', linewidth=1.0, alpha=0.8)
    ax2.axhline(y=0, color='#7f8c8d', linewidth=0.5, linestyle='-')

    # Mark peak force (max acceleration) per rep
    if rep_ranges:
        for i, (start, peak, end) in enumerate(rep_ranges):
            s = max(0, start)
            e = min(len(acceleration) - 1, end)
            rep_accel = acceleration[s:e + 1]
            if len(rep_accel) > 0:
                peak_accel_idx = s + np.argmax(rep_accel)
                if peak_accel_idx < len(times):
                    ax2.scatter([times[peak_accel_idx]], [acceleration[peak_accel_idx]],
                                color='#e74c3c', s=40, zorder=5)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (px/s²)")
    ax2.set_title("Bar Acceleration / 杠铃加速度（发力曲线）", fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
