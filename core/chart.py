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
