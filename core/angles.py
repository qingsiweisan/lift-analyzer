"""Angle calculation utilities for pose analysis."""

import math
import numpy as np


def calculate_angle(a, b, c):
    """Calculate the angle at point b formed by points a-b-c.

    Args:
        a, b, c: Each is (x, y) tuple or array.

    Returns:
        Angle in degrees (0-180).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def spine_inclination(shoulder, hip):
    """Calculate the forward lean angle of the spine relative to vertical.

    Args:
        shoulder: (x, y) of the shoulder.
        hip: (x, y) of the hip.

    Returns:
        Angle in degrees. 0 = perfectly upright, 90 = horizontal.
    """
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]  # image y-axis is inverted
    return abs(math.degrees(math.atan2(dx, -dy)))


def horizontal_offset(pt_a, pt_b):
    """Return the horizontal pixel distance between two points."""
    return abs(pt_a[0] - pt_b[0])


def vertical_offset(pt_a, pt_b):
    """Return the vertical pixel distance between two points."""
    return abs(pt_a[1] - pt_b[1])


def symmetry_ratio(left_val, right_val):
    """Return symmetry ratio (0-1, 1 = perfectly symmetric)."""
    max_val = max(abs(left_val), abs(right_val))
    if max_val < 1e-6:
        return 1.0
    return min(abs(left_val), abs(right_val)) / max_val
