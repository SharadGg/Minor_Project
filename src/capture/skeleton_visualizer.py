"""Skeleton Visualization Utilities"""
import cv2
import numpy as np
from typing import Tuple, List

def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    connections: List[Tuple[int, int]] = None,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    line_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2
) -> np.ndarray:
    """Draw skeleton on image."""
    output = image.copy()
    h, w = image.shape[:2]
    
    if connections is None:
        connections = get_mediapipe_connections()
    
    # Draw connections (bones)
    for parent, child in connections:
        if parent < len(keypoints) and child < len(keypoints):
            conf_parent = keypoints[parent, 2] if keypoints.shape[1] > 2 else 1.0
            conf_child = keypoints[child, 2] if keypoints.shape[1] > 2 else 1.0
            
            if conf_parent > 0.5 and conf_child > 0.5:
                x1, y1 = int(keypoints[parent, 0] * w), int(keypoints[parent, 1] * h)
                x2, y2 = int(keypoints[child, 0] * w), int(keypoints[child, 1] * h)
                cv2.line(output, (x1, y1), (x2, y2), line_color, thickness)
    
    # Draw joints
    for i, point in enumerate(keypoints):
        confidence = point[2] if len(point) > 2 else 1.0
        if confidence > 0.5:
            x, y = int(point[0] * w), int(point[1] * h)
            cv2.circle(output, (x, y), 4, point_color, -1)
            cv2.circle(output, (x, y), 4, (255, 255, 255), 1)
    
    return output

def get_mediapipe_connections() -> List[Tuple[int, int]]:
    """Get MediaPipe pose connections."""
    return [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
        (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
    ]