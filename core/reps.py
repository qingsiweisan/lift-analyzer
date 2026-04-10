"""Rep counting via peak detection in angle data."""

import numpy as np


def find_peaks(data, min_prominence=20, min_distance=15):
    """Find peaks in a 1D signal. Simple implementation without scipy.

    Args:
        data: list or array of values.
        min_prominence: minimum height difference from surrounding valleys.
        min_distance: minimum number of samples between peaks.

    Returns:
        List of peak indices.
    """
    data = np.array(data, dtype=float)
    n = len(data)
    if n < 3:
        return []

    # Smooth the data to reduce noise (simple moving average)
    kernel = max(3, min(15, n // 20))
    if kernel % 2 == 0:
        kernel += 1
    smoothed = np.convolve(data, np.ones(kernel) / kernel, mode='same')

    # Find local maxima
    candidates = []
    for i in range(1, n - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            candidates.append(i)

    if not candidates:
        return []

    # Filter by prominence
    peaks = []
    for idx in candidates:
        # Find left valley
        left_min = smoothed[idx]
        for j in range(idx - 1, max(0, idx - n // 2) - 1, -1):
            left_min = min(left_min, smoothed[j])
            if smoothed[j] > smoothed[idx]:
                break

        # Find right valley
        right_min = smoothed[idx]
        for j in range(idx + 1, min(n, idx + n // 2)):
            right_min = min(right_min, smoothed[j])
            if smoothed[j] > smoothed[idx]:
                break

        prominence = smoothed[idx] - max(left_min, right_min)
        if prominence >= min_prominence:
            peaks.append((idx, smoothed[idx], prominence))

    if not peaks:
        return []

    # Filter by minimum distance (keep the most prominent in each cluster)
    peaks.sort(key=lambda x: x[0])
    filtered = [peaks[0]]
    for p in peaks[1:]:
        if p[0] - filtered[-1][0] >= min_distance:
            filtered.append(p)
        elif p[2] > filtered[-1][2]:  # higher prominence
            filtered[-1] = p

    return [p[0] for p in filtered]


def find_valleys(data, min_prominence=20, min_distance=15):
    """Find valleys (local minima) by inverting and finding peaks."""
    inverted = [-x for x in data]
    return find_peaks(inverted, min_prominence, min_distance)


def detect_reps(angle_data, mode="peak", min_prominence=20, min_distance_sec=1.0, fps=30):
    """Detect individual reps from an angle time series.

    Args:
        angle_data: list of angle values per frame.
        mode: "peak" to count lockout peaks, "valley" to count bottom valleys.
        min_prominence: minimum angle change to count as a rep.
        min_distance_sec: minimum time between reps in seconds.
        fps: video framerate.

    Returns:
        dict with:
            "count": number of reps
            "peak_frames": list of frame indices at peak/valley of each rep
            "rep_ranges": list of (start_frame, peak_frame, end_frame) per rep
    """
    min_distance = int(min_distance_sec * fps)

    if mode == "peak":
        rep_frames = find_peaks(angle_data, min_prominence, min_distance)
    else:
        rep_frames = find_valleys(angle_data, min_prominence, min_distance)

    # Determine rep boundaries: midpoint between consecutive peaks
    rep_ranges = []
    for i, peak in enumerate(rep_frames):
        if i == 0:
            start = 0
        else:
            start = (rep_frames[i - 1] + peak) // 2

        if i == len(rep_frames) - 1:
            end = len(angle_data) - 1
        else:
            end = (peak + rep_frames[i + 1]) // 2

        rep_ranges.append((start, peak, end))

    return {
        "count": len(rep_frames),
        "peak_frames": rep_frames,
        "rep_ranges": rep_ranges,
    }
