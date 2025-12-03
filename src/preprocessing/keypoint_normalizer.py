"""Keypoint Normalization"""
import numpy as np

def normalize_skeleton(keypoints: np.ndarray, center_joint: int = 0, scale_method: str = 'std') -> np.ndarray:
    """Normalize skeleton keypoints."""
    normalized = keypoints.copy()
    coords = keypoints[:, :2]
    
    # Center at specified joint
    center = coords[center_joint]
    coords = coords - center
    
    # Scale
    if scale_method == 'std':
        scale = np.std(coords) + 1e-8
    else:
        scale = np.max(np.abs(coords)) + 1e-8
    coords = coords / scale
    
    normalized[:, :2] = coords
    return normalized

def compute_joint_angles(keypoints: np.ndarray) -> dict:
    """Compute angles at major joints."""
    def angle(p1, p2, p3):
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return {
        'left_elbow': angle(keypoints[11], keypoints[13], keypoints[15]),
        'right_elbow': angle(keypoints[12], keypoints[14], keypoints[16]),
        'left_knee': angle(keypoints[23], keypoints[25], keypoints[27]),
        'right_knee': angle(keypoints[24], keypoints[26], keypoints[28]),
    }