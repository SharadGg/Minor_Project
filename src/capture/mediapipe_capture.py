"""
MediaPipe Pose Extraction Module

Extracts 2D and 3D pose keypoints from images/video using MediaPipe BlazePose.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional


class MediaPipePoseExtractor:
    """
    Extract pose keypoints using MediaPipe BlazePose.
    
    MediaPipe provides 33 3D landmarks with (x, y, z, visibility):
    - x, y: Normalized to [0, 1]
    - z: Depth relative to hips
    - visibility: Confidence score [0, 1]
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,  # 0, 1, or 2 (higher = more accurate but slower)
        smooth_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe Pose.
        
        Args:
            static_image_mode: If True, treat each image independently
            model_complexity: 0 (lite), 1 (full), or 2 (heavy)
            smooth_landmarks: If True, smooths landmarks across frames
            min_detection_confidence: Min confidence for pose detection
            min_tracking_confidence: Min confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Define landmark indices
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # Key body part groups
        self.body_parts = {
            'face': list(range(0, 11)),
            'upper_body': [11, 12, 13, 14, 15, 16, 23, 24],
            'arms': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'lower_body': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        }
    
    def extract_pose(
        self,
        image: np.ndarray,
        draw_landmarks: bool = True
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Extract pose from single image/frame.
        
        Args:
            image: Input image (BGR format)
            draw_landmarks: If True, draw skeleton on image
        
        Returns:
            keypoints: (33, 4) array of [x, y, z, visibility], None if no pose detected
            annotated_image: Image with drawn skeleton
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        # Create annotated image
        annotated_image = image.copy()
        
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = self._landmarks_to_array(results.pose_landmarks)
            
            # Draw landmarks if requested
            if draw_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            return keypoints, annotated_image
        else:
            return None, annotated_image
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """
        Convert MediaPipe landmarks to numpy array.
        
        Args:
            landmarks: MediaPipe pose landmarks
        
        Returns:
            Array of shape (33, 4) with [x, y, z, visibility]
        """
        keypoints = np.zeros((33, 4))
        
        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i] = [
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ]
        
        return keypoints
    
    def extract_2d_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract only 2D coordinates (x, y) with visibility.
        
        Args:
            keypoints: (33, 4) array
        
        Returns:
            (33, 3) array of [x, y, visibility]
        """
        return keypoints[:, [0, 1, 3]]
    
    def extract_3d_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract 3D coordinates (x, y, z).
        
        Args:
            keypoints: (33, 4) array
        
        Returns:
            (33, 3) array of [x, y, z]
        """
        return keypoints[:, :3]
    
    def filter_keypoints_by_visibility(
        self,
        keypoints: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Filter out keypoints with low visibility.
        
        Args:
            keypoints: (33, 4) array
            threshold: Min visibility threshold
        
        Returns:
            Filtered keypoints with low-visibility points set to zero
        """
        filtered = keypoints.copy()
        low_visibility_mask = keypoints[:, 3] < threshold
        filtered[low_visibility_mask, :3] = 0
        return filtered
    
    def get_body_part_keypoints(
        self,
        keypoints: np.ndarray,
        body_part: str
    ) -> np.ndarray:
        """
        Get keypoints for specific body part.
        
        Args:
            keypoints: (33, 4) array
            body_part: 'face', 'upper_body', 'arms', or 'lower_body'
        
        Returns:
            Keypoints for specified body part
        """
        if body_part not in self.body_parts:
            raise ValueError(f"Unknown body part: {body_part}")
        
        indices = self.body_parts[body_part]
        return keypoints[indices]
    
    def calculate_angles(self, keypoints: np.ndarray) -> dict:
        """
        Calculate important joint angles.
        
        Args:
            keypoints: (33, 4) array
        
        Returns:
            Dictionary of angles in degrees
        """
        def angle_between_points(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3."""
            v1 = p1[:2] - p2[:2]
            v2 = p3[:2] - p2[:2]
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        angles = {}
        
        # Left elbow angle
        angles['left_elbow'] = angle_between_points(
            keypoints[11], keypoints[13], keypoints[15]  # shoulder, elbow, wrist
        )
        
        # Right elbow angle
        angles['right_elbow'] = angle_between_points(
            keypoints[12], keypoints[14], keypoints[16]
        )
        
        # Left knee angle
        angles['left_knee'] = angle_between_points(
            keypoints[23], keypoints[25], keypoints[27]  # hip, knee, ankle
        )
        
        # Right knee angle
        angles['right_knee'] = angle_between_points(
            keypoints[24], keypoints[26], keypoints[28]
        )
        
        # Left shoulder angle
        angles['left_shoulder'] = angle_between_points(
            keypoints[23], keypoints[11], keypoints[13]  # hip, shoulder, elbow
        )
        
        # Right shoulder angle
        angles['right_shoulder'] = angle_between_points(
            keypoints[24], keypoints[12], keypoints[14]
        )
        
        return angles
    
    def estimate_body_orientation(self, keypoints: np.ndarray) -> str:
        """
        Estimate if person is facing camera, left, or right.
        
        Args:
            keypoints: (33, 4) array
        
        Returns:
            'front', 'left', or 'right'
        """
        # Compare visibility of left and right shoulders
        left_shoulder_vis = keypoints[11, 3]
        right_shoulder_vis = keypoints[12, 3]
        
        # Compare x-coordinates of nose and shoulders
        nose_x = keypoints[0, 0]
        left_shoulder_x = keypoints[11, 0]
        right_shoulder_x = keypoints[12, 0]
        
        # Determine orientation
        if abs(left_shoulder_vis - right_shoulder_vis) < 0.2:
            return 'front'
        elif left_shoulder_vis > right_shoulder_vis:
            return 'left'
        else:
            return 'right'
    
    def close(self):
        """Clean up resources."""
        self.pose.close()


def process_video(
    video_path: str,
    output_path: str = None,
    show_realtime: bool = True
) -> list:
    """
    Process entire video and extract pose keypoints.
    
    Args:
        video_path: Path to input video
        output_path: Path to save annotated video (optional)
        show_realtime: Show video while processing
    
    Returns:
        List of keypoints for each frame
    """
    pose_extractor = MediaPipePoseExtractor()
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup video writer if output path provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_keypoints = []
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract pose
            keypoints, annotated_frame = pose_extractor.extract_pose(frame)
            
            if keypoints is not None:
                all_keypoints.append(keypoints)
            else:
                all_keypoints.append(None)
            
            # Save annotated frame
            if writer:
                writer.write(annotated_frame)
            
            # Display
            if show_realtime:
                cv2.imshow('Processing', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        pose_extractor.close()
    
    print(f"Processing complete. Extracted {len([k for k in all_keypoints if k is not None])} valid poses.")
    
    return all_keypoints


if __name__ == "__main__":
    # Test on webcam
    print("Testing MediaPipe Pose Extractor on webcam...")
    print("Press 'q' to quit")
    
    pose_extractor = MediaPipePoseExtractor()
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            keypoints, annotated_frame = pose_extractor.extract_pose(frame)
            
            if keypoints is not None:
                # Calculate angles
                angles = pose_extractor.calculate_angles(keypoints)
                orientation = pose_extractor.estimate_body_orientation(keypoints)
                
                # Display info
                cv2.putText(annotated_frame, f"Orientation: {orientation}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Left Elbow: {angles['left_elbow']:.1f}°", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('MediaPipe Pose', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_extractor.close()