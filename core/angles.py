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


# --- 3D angle calculations (for WHAM backend) ---

def calculate_angle_3d(a, b, c):
    """Calculate the angle at point b formed by 3D points a-b-c.

    Args:
        a, b, c: Each is (x, y, z) tuple or array.

    Returns:
        Angle in degrees (0-180).
    """
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def spine_inclination_3d(shoulder, hip):
    """Calculate the forward lean angle of the spine in 3D space.

    Uses the full 3D vector from hip to shoulder vs the vertical axis.
    SMPL convention: Y-axis is vertical, negative Y = up.

    Args:
        shoulder: (x, y, z) of the shoulder.
        hip: (x, y, z) of the hip.

    Returns:
        Angle in degrees. 0 = perfectly upright, 90 = horizontal.
    """
    spine = np.array(shoulder[:3]) - np.array(hip[:3])
    vertical = np.array([0.0, -1.0, 0.0])  # up direction (SMPL: negative Y = up)
    cosine = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def is_3d(point):
    """Check if a point has 3D coordinates (len >= 3 and z != 0)."""
    return len(point) >= 3 and point[2] != 0


def auto_angle(a, b, c):
    """Automatically choose 2D or 3D angle calculation based on input dimension."""
    if is_3d(a) and is_3d(b) and is_3d(c):
        return calculate_angle_3d(a, b, c)
    return calculate_angle(a[:2], b[:2], c[:2])


def auto_spine_inclination(shoulder, hip):
    """Automatically choose 2D or 3D spine inclination based on input dimension."""
    if is_3d(shoulder) and is_3d(hip):
        return spine_inclination_3d(shoulder, hip)
    return spine_inclination(shoulder[:2], hip[:2])
