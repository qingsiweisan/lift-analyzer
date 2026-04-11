"""WHAM 3D pose estimation adapter.

Wraps WHAM (https://github.com/yohanshin/WHAM) to extract per-frame 3D joint
coordinates from a video, mapping SMPL 24-joint output to our semantic keypoint
names (shoulder, hip, knee, ankle, wrist, elbow, etc.).

Requires:
    - WHAM installed and importable (see WHAM/docs/INSTALL.md)
    - SMPL model files downloaded
    - CUDA GPU with >= 8GB VRAM
"""

import os
import sys
import numpy as np

# SMPL 24-joint indices → our keypoint names
# Reference: https://meshcapade.wiki/SMPL#skeleton-layout
SMPL_JOINT_MAP = {
    "hip_center": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine1": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine2": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_shoulder": 13,
    "right_shoulder": 14,
    "head": 15,
    "left_elbow": 16,
    "right_elbow": 17,
    "left_wrist": 18,
    "right_wrist": 19,
    "left_hand": 20,
    "right_hand": 21,
    "left_heel": 22,  # WHAM extended
    "right_heel": 23,
}

# Map our semantic names to SMPL joint indices (per side)
KEYPOINT_MAP_LEFT = {
    "shoulder": SMPL_JOINT_MAP["left_shoulder"],
    "elbow": SMPL_JOINT_MAP["left_elbow"],
    "wrist": SMPL_JOINT_MAP["left_wrist"],
    "hip": SMPL_JOINT_MAP["left_hip"],
    "knee": SMPL_JOINT_MAP["left_knee"],
    "ankle": SMPL_JOINT_MAP["left_ankle"],
    "heel": SMPL_JOINT_MAP["left_heel"],
    "ear": SMPL_JOINT_MAP["head"],  # approximate: SMPL has no ear
    "foot_index": SMPL_JOINT_MAP["left_foot"],
}

KEYPOINT_MAP_RIGHT = {
    "shoulder": SMPL_JOINT_MAP["right_shoulder"],
    "elbow": SMPL_JOINT_MAP["right_elbow"],
    "wrist": SMPL_JOINT_MAP["right_wrist"],
    "hip": SMPL_JOINT_MAP["right_hip"],
    "knee": SMPL_JOINT_MAP["right_knee"],
    "ankle": SMPL_JOINT_MAP["right_ankle"],
    "heel": SMPL_JOINT_MAP["right_heel"],
    "ear": SMPL_JOINT_MAP["head"],
    "foot_index": SMPL_JOINT_MAP["right_foot"],
}


class WHAMPoseExtractor:
    """Extract 3D joint positions from video using WHAM.

    WHAM runs inside WSL Ubuntu (requires conda env 'wham' with CUDA GPU).
    This class handles the Windows → WSL bridge: copies the video in, runs
    WHAM inference, extracts 24 SMPL joints from mesh vertices, and returns
    the result as a numpy array.

    Usage:
        extractor = WHAMPoseExtractor()
        result = extractor.process(video_path)
        # result["joints_3d"]: np.array (num_frames, 24, 3)
    """

    WHAM_DIR = "/root/WHAM"
    CONDA_ENV = "wham"

    def __init__(self, wham_dir=None):
        self.wham_dir = wham_dir or os.environ.get("WHAM_DIR", self.WHAM_DIR)

    def process(self, video_path):
        """Run WHAM on a video and return 3D joint positions.

        Args:
            video_path: Path to input video (Windows path).

        Returns:
            dict with:
                "joints_3d": np.array (num_frames, 24, 3) — 3D joint coords in meters
                "frame_ids": np.array (num_frames,) — original frame indices
                "fps": float — video FPS
                "num_frames": int
                "total_video_frames": int — total frames in the video
        """
        import subprocess
        import tempfile

        video_path = os.path.abspath(video_path)
        wsl_video = self._win_to_wsl(video_path)

        # Temp file for numpy output (WSL-accessible via /mnt/)
        tmp_dir = tempfile.mkdtemp(prefix="wham_")
        joints_npy = os.path.join(tmp_dir, "joints_3d.npy")
        frame_ids_npy = os.path.join(tmp_dir, "frame_ids.npy")
        wsl_joints = self._win_to_wsl(joints_npy)
        wsl_frame_ids = self._win_to_wsl(frame_ids_npy)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        wham_output_dir = f"{self.wham_dir}/output/lift_analyzer"
        pkl_path = f"{wham_output_dir}/{video_name}/wham_output.pkl"

        # Step 1: Run WHAM demo.py (skip if output already exists)
        check_result = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "-e", "test", "-f", pkl_path],
            capture_output=True, timeout=10,
        )
        if check_result.returncode == 0:
            print(f"  WHAM output cached, skipping inference.")
        else:
            print(f"  Running WHAM inference (WSL)...")
            demo_cmd = (
                f"cd {self.wham_dir} && "
                f"/root/miniconda3/bin/conda run --no-capture-output -n {self.CONDA_ENV} "
                f"python demo.py --video {wsl_video} "
                f"--output_pth {wham_output_dir} --save_pkl --estimate_local_only"
            )
            result = subprocess.run(
                ["wsl.exe", "-d", "Ubuntu", "-e", "sudo", "bash", "-c", demo_cmd],
                capture_output=True, text=True, timeout=1200,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"WHAM demo.py failed (exit {result.returncode}):\n"
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            print(f"  WHAM inference complete.")

        # Step 2: Extract joints from WHAM output
        extract_script = self._build_extract_script(pkl_path, wsl_joints, wsl_frame_ids)
        result = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "-e", "sudo", "bash", "-c",
             f"cd {self.wham_dir} && "
             f"/root/miniconda3/bin/conda run --no-capture-output -n {self.CONDA_ENV} "
             f"python -c {self._shell_quote(extract_script)}"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Joint extraction failed:\n{result.stderr.strip() or result.stdout.strip()}"
            )

        # Load results
        if not os.path.exists(joints_npy):
            raise RuntimeError("WHAM produced no output. Check WSL/CUDA setup.")

        joints_3d = np.load(joints_npy)
        frame_ids = np.load(frame_ids_npy)

        # Get video FPS
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print(f"  WHAM: {joints_3d.shape[0]} frames processed, "
              f"{total_frames} total video frames, {fps:.1f} FPS")

        return {
            "joints_3d": joints_3d,
            "frame_ids": frame_ids,
            "num_frames": joints_3d.shape[0],
            "total_video_frames": total_frames,
            "fps": fps,
        }

    def _build_extract_script(self, pkl_path, wsl_joints_out, wsl_frame_ids_out):
        """Build Python script to extract joints from WHAM pkl output."""
        return f"""
import numpy as np, joblib, sys, os
os.chdir('{self.wham_dir}')
sys.path.insert(0, '{self.wham_dir}')
results = joblib.load('{pkl_path}')
sid = max(results.keys(), key=lambda k: results[k]['verts'].shape[0])
data = results[sid]
from lib.models.smpl import SMPL
smpl = SMPL('{self.wham_dir}/dataset/body_models/smpl/', gender='neutral', batch_size=1)
j_reg = smpl.J_regressor.numpy()
joints_3d = np.einsum('jv,nvd->njd', j_reg, data['verts'])
np.save('{wsl_joints_out}', joints_3d.astype(np.float32))
np.save('{wsl_frame_ids_out}', data['frame_ids'])
"""

    @staticmethod
    def _win_to_wsl(win_path):
        """Convert Windows path to WSL /mnt/ path."""
        path = win_path.replace("\\", "/")
        if len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            return f"/mnt/{drive}{path[2:]}"
        return path

    @staticmethod
    def _shell_quote(s):
        """Quote a string for bash -c '...' usage."""
        return "'" + s.replace("'", "'\\''") + "'"


def extract_points_3d(joints_3d_frame, side="left"):
    """Extract named 3D keypoints from a single frame's SMPL joints.

    Args:
        joints_3d_frame: np.array of shape (24, 3) — one frame of SMPL joints.
        side: "left" or "right"

    Returns:
        dict of name -> (x, y, z) tuples
    """
    kmap = KEYPOINT_MAP_LEFT if side == "left" else KEYPOINT_MAP_RIGHT
    points = {}
    for name, idx in kmap.items():
        if idx < len(joints_3d_frame):
            points[name] = tuple(joints_3d_frame[idx].tolist())
        else:
            points[name] = (0.0, 0.0, 0.0)
    return points


def extract_both_sides_3d(joints_3d_frame):
    """Extract 3D keypoints for both sides from a single frame.

    Returns:
        {"left": {name: (x,y,z), ...}, "right": {name: (x,y,z), ...}}
    """
    return {
        "left": extract_points_3d(joints_3d_frame, "left"),
        "right": extract_points_3d(joints_3d_frame, "right"),
    }


def pick_visible_side_3d(joints_3d_frame, camera_direction=None):
    """Pick the side closer to the camera.

    For 3D data, we use the Z-coordinate (depth) of the hips.
    The side with the hip closer to the camera (smaller Z in typical convention)
    is considered the "visible" side.

    Args:
        joints_3d_frame: np.array (24, 3)
        camera_direction: not used for 3D (kept for API compat)

    Returns:
        ("left", KEYPOINT_MAP_LEFT) or ("right", KEYPOINT_MAP_RIGHT)
    """
    left_hip_z = joints_3d_frame[SMPL_JOINT_MAP["left_hip"]][2]
    right_hip_z = joints_3d_frame[SMPL_JOINT_MAP["right_hip"]][2]

    # Smaller Z = closer to camera (standard convention)
    if left_hip_z <= right_hip_z:
        return "left", KEYPOINT_MAP_LEFT
    return "right", KEYPOINT_MAP_RIGHT
