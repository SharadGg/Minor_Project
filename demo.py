"""
Real-Time Pose-based Gesture and Emotion Recognition Demo
CPU OPTIMIZED for Intel i3, 8GB RAM, No GPU

Location: S:\\Everything\\Clg_Project\\Final_Draft\\PoseGestureAnalyzer\\demo.py
Features:
- Real-time webcam capture
- MediaPipe pose extraction
- CPU inference (3-5 FPS expected)
- Skeleton visualization
- Live predictions with confidence
- Optimized for low-end hardware
"""

import cv2
import torch
import numpy as np
from collections import deque
import argparse
import yaml
import time
from pathlib import Path
import sys

# Import custom modules
from src.models.model_factory import create_model
from src.capture.mediapipe_capture import MediaPipePoseExtractor
from src.preprocessing.keypoint_normalizer import normalize_skeleton
from src.capture.skeleton_visualizer import draw_skeleton
from src.utils.checkpoint import load_checkpoint


class RealtimeGestureRecognizer:
    """
    Real-time gesture and emotion recognition system.
    Optimized for CPU-only inference.
    """
    def __init__(self, config_path, checkpoint_path):
        print("\n" + "="*60)
        print("🎬 INITIALIZING POSE GESTURE RECOGNITION DEMO")
        print("="*60)
        
        # Load configuration
        print("🔧 Loading configuration...")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Force CPU device
        self.device = torch.device('cpu')
        print(f"🖥️  Device: CPU (Optimized for Intel i3)")
        
        # Class labels
        self.emotion_labels = ['Happy', 'Sad', 'Angry', 'Confused', 'Neutral', 'Excited', 'Fearful']
        self.gesture_labels = ['Waving', 'Pointing', 'Asking', 'Signaling', 'Warning', 
                              'Greeting', 'Dismissing', 'Celebrating']
        self.all_labels = self.emotion_labels + self.gesture_labels
        
        print(f"📊 Classes: {len(self.all_labels)} ({len(self.emotion_labels)} emotions + {len(self.gesture_labels)} gestures)")
        
        # Load model
        print("🧠 Loading model...")
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        print("✓ Model loaded successfully")
        
        # Initialize MediaPipe
        print("📷 Initializing MediaPipe...")
        self.pose_extractor = MediaPipePoseExtractor()
        print("✓ MediaPipe initialized")
        
        # Sliding window for temporal sequences
        self.window_size = self.config['inference']['window_size']
        self.sequence_buffer = deque(maxlen=self.window_size)
        print(f"⏱️  Window size: {self.window_size} frames")
        
        # Performance tracking
        self.fps_buffer = deque(maxlen=30)
        
        # Confidence threshold
        self.confidence_threshold = self.config['inference']['confidence_threshold']
        
        # Temporal smoothing
        self.prediction_history = deque(maxlen=5)
        
        print("\n✅ Initialization complete!")
        print("="*60 + "\n")
        
    def _load_model(self, checkpoint_path):
        """Load trained model from checkpoint."""
        # Create model
        model = create_model(
            model_name=self.config['model']['name'],
            num_classes=len(self.all_labels),
            num_joints=self.config['model']['num_joints'],
            **self.config['model']['params']
        )
        
        # Load weights
        checkpoint = load_checkpoint(checkpoint_path)
        
        # Handle DataParallel wrapper
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        print(f"  ↳ Checkpoint epoch: {checkpoint['epoch']}")
        print(f"  ↳ Best val accuracy: {checkpoint['best_val_acc']:.4f}")
        
        return model
    
    def preprocess_keypoints(self, keypoints):
        """
        Preprocess keypoints for model input.
        
        Args:
            keypoints: (num_joints, 3) array of (x, y, confidence)
        
        Returns:
            Normalized keypoints
        """
        # Normalize skeleton
        normalized = normalize_skeleton(keypoints)
        return normalized
    
    def predict(self, sequence):
        """
        Make prediction on sequence.
        
        Args:
            sequence: List of keypoint frames
        
        Returns:
            predicted_class, confidence, probabilities
        """
        if len(sequence) < self.window_size:
            return None, 0.0, None
        
        # Convert to tensor
        sequence_array = np.array(sequence)
        
        # Reshape to model input format: (1, channels, frames, joints, persons)
        sequence_tensor = torch.FloatTensor(sequence_array).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(sequence_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        probs = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, probs
    
    def smooth_predictions(self, prediction, confidence):
        """Apply temporal smoothing to predictions."""
        if prediction is None:
            return None, 0.0
        
        self.prediction_history.append((prediction, confidence))
        
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Vote for most common prediction
        predictions = [p for p, c in self.prediction_history]
        confidences = [c for p, c in self.prediction_history]
        
        # Most common prediction
        most_common = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean([c for p, c in self.prediction_history if p == most_common])
        
        return most_common, avg_confidence
    
    def draw_info(self, frame, prediction, confidence, fps):
        """Draw prediction information on frame."""
        h, w = frame.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw title
        cv2.putText(frame, "Pose Gesture Recognition", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Draw FPS
        fps_color = (0, 255, 0) if fps >= 3 else (0, 255, 255)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)
        
        # Draw prediction
        if prediction is not None and confidence >= self.confidence_threshold:
            label = self.all_labels[prediction]
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
                conf_text = "HIGH"
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow
                conf_text = "MEDIUM"
            else:
                color = (0, 165, 255)  # Orange
                conf_text = "LOW"
            
            cv2.putText(frame, f"Prediction: {label}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%} ({conf_text})", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Detecting gesture...", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(frame, f"Buffer: {len(self.sequence_buffer)}/{self.window_size}", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return frame
    
    def draw_top_predictions(self, frame, probabilities):
        """Draw top-3 predictions with bar chart."""
        if probabilities is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Get top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Draw on right side
        x_start = w - 350
        y_start = 220
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 10, y_start - 40), (w - 10, y_start + 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Title
        cv2.putText(frame, "Top 3 Predictions:", (x_start, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, idx in enumerate(top3_indices):
            label = self.all_labels[idx]
            prob = probabilities[idx]
            
            y_pos = y_start + i * 70
            
            # Draw label
            cv2.putText(frame, f"{i+1}. {label}", (x_start, y_pos + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw probability bar
            bar_length = int(250 * prob)
            color = (0, 255, 0) if i == 0 else (100, 200, 255) if i == 1 else (150, 150, 150)
            cv2.rectangle(frame, (x_start, y_pos + 30), 
                         (x_start + bar_length, y_pos + 50), color, -1)
            cv2.rectangle(frame, (x_start, y_pos + 30), 
                         (x_start + 250, y_pos + 50), (255, 255, 255), 1)
            
            # Draw percentage
            cv2.putText(frame, f"{prob:.1%}", (x_start + bar_length + 10, y_pos + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_instructions(self, frame):
        """Draw control instructions."""
        h, w = frame.shape[:2]
        
        instructions = [
            "Controls:",
            "Q - Quit",
            "R - Reset",
            "S - Screenshot"
        ]
        
        y_start = h - 130
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, y_start + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Run real-time demo."""
        print("\n" + "="*60)
        print("🎬 STARTING REAL-TIME DEMO")
        print("="*60)
        print("\n📹 Opening webcam...")
        print("\nControls:")
        print("  'Q' - Quit")
        print("  'R' - Reset buffer")
        print("  'S' - Save screenshot")
        print("\n" + "="*60 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ ERROR: Could not open webcam")
            print("   Please check:")
            print("   1. Webcam is connected")
            print("   2. No other app is using webcam")
            print("   3. Camera permissions are granted")
            return
        
        print("✅ Webcam opened successfully!\n")
        print("⏳ Buffering frames... Please perform gestures slowly.\n")
        
        last_prediction = None
        last_confidence = 0.0
        last_probabilities = None
        frame_count = 0
        screenshot_count = 0
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Extract pose
                keypoints, annotated_frame = self.pose_extractor.extract_pose(frame)
                
                if keypoints is not None:
                    # Preprocess keypoints
                    processed_keypoints = self.preprocess_keypoints(keypoints)
                    
                    # Add to buffer
                    self.sequence_buffer.append(processed_keypoints)
                    
                    # Make prediction if buffer is full
                    if len(self.sequence_buffer) == self.window_size:
                        prediction, confidence, probabilities = self.predict(list(self.sequence_buffer))
                        
                        if prediction is not None:
                            # Apply temporal smoothing
                            prediction, confidence = self.smooth_predictions(prediction, confidence)
                            
                            if confidence >= self.confidence_threshold:
                                last_prediction = prediction
                                last_confidence = confidence
                                last_probabilities = probabilities
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time + 1e-6)
                self.fps_buffer.append(fps)
                avg_fps = np.mean(self.fps_buffer)
                
                # Draw visualizations
                annotated_frame = self.draw_info(annotated_frame, last_prediction, last_confidence, avg_fps)
                annotated_frame = self.draw_top_predictions(annotated_frame, last_probabilities)
                annotated_frame = self.draw_instructions(annotated_frame)
                
                # Show frame
                cv2.imshow('Pose Gesture Recognition - CPU Demo', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n🛑 Quitting...")
                    break
                
                elif key == ord('r') or key == ord('R'):
                    print("♻️  Buffer reset")
                    self.sequence_buffer.clear()
                    self.prediction_history.clear()
                    last_prediction = None
                    last_confidence = 0.0
                    last_probabilities = None
                
                elif key == ord('s') or key == ord('S'):
                    filename = f"screenshot_{screenshot_count:03d}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    screenshot_count += 1
                    print(f"📸 Screenshot saved: {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.pose_extractor.close()
            
            print("\n" + "="*60)
            print("📊 DEMO STATISTICS")
            print("="*60)
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {np.mean(self.fps_buffer):.2f}")
            print(f"Screenshots saved: {screenshot_count}")
            print("="*60 + "\n")
            print("✅ Demo ended successfully!\n")


def main():
    parser = argparse.ArgumentParser(description='Real-time Gesture Recognition Demo (CPU Optimized)')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.config).exists():
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run demo
    recognizer = RealtimeGestureRecognizer(
        args.config,
        args.checkpoint
    )
    recognizer.run()


if __name__ == '__main__':
    main()