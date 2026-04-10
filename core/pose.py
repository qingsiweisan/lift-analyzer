"""MediaPipe Pose wrapper for landmark extraction."""

import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# Landmark groups for convenience
LANDMARKS_LEFT = {
    "ear": mp_pose.PoseLandmark.LEFT_EAR,
    "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "hip": mp_pose.PoseLandmark.LEFT_HIP,
    "knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "heel": mp_pose.PoseLandmark.LEFT_HEEL,
    "foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
}

LANDMARKS_RIGHT = {
    "ear": mp_pose.PoseLandmark.RIGHT_EAR,
    "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "heel": mp_pose.PoseLandmark.RIGHT_HEEL,
    "foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
}


def create_pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Create a MediaPipe Pose instance."""
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def get_coords(landmarks, idx, width, height):
    """Get (x, y) pixel coordinates for a landmark index.

    Returns:
        ((x, y), visibility)
    """
    lm = landmarks[idx]
    return (lm.x * width, lm.y * height), lm.visibility


def pick_visible_side(landmarks):
    """Pick the side (left/right) with higher hip visibility.

    Returns:
        ("left", LANDMARKS_LEFT) or ("right", LANDMARKS_RIGHT)
    """
    left_vis = landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility
    right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility
    if left_vis >= right_vis:
        return "left", LANDMARKS_LEFT
    return "right", LANDMARKS_RIGHT


def extract_points(landmarks, side_landmarks, width, height):
    """Extract all named points for a given side.

    Returns:
        dict of name -> (x, y) pixel coords
    """
    points = {}
    for name, idx in side_landmarks.items():
        coords, _ = get_coords(landmarks, idx, width, height)
        points[name] = coords
    return points


def extract_both_sides(landmarks, width, height):
    """Extract points for both left and right sides.

    Returns:
        {"left": {name: (x,y), ...}, "right": {name: (x,y), ...}}
    """
    return {
        "left": extract_points(landmarks, LANDMARKS_LEFT, width, height),
        "right": extract_points(landmarks, LANDMARKS_RIGHT, width, height),
    }


def draw_pose(frame, pose_landmarks):
    """Draw pose landmarks and connections on a frame."""
    mp_drawing.draw_landmarks(
        frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2),
    )
