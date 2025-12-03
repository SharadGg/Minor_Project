"""
Generate Synthetic Skeleton Data for Testing
Creates fake but valid skeleton sequences for quick testing without real data

Location: S:\Everything\Clg_Project\Final_Draft\PoseGestureAnalyzer\datasets\generate_sample_data.py
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import os


def generate_sample_sequence(
    num_frames: int = 150,  # Reduced from 300 for CPU
    num_joints: int = 25,
    gesture_type: int = 0
) -> np.ndarray:
    """
    Generate a synthetic skeleton sequence.
    
    Args:
        num_frames: Number of frames in sequence
        num_joints: Number of skeleton joints
        gesture_type: Class ID (0-14)
    
    Returns:
        Sequence array of shape (num_frames, num_joints, 3)
    """
    
    # Base pose (standing neutral)
    base_pose = np.zeros((num_joints, 3))
    
    # Define basic skeleton structure (normalized coordinates)
    # Joint positions for standing pose
    base_pose[0] = [0.0, 0.0, 0.9]    # Nose
    base_pose[11] = [-0.2, 0.0, 0.6]  # Left shoulder
    base_pose[12] = [0.2, 0.0, 0.6]   # Right shoulder
    base_pose[13] = [-0.3, 0.0, 0.4]  # Left elbow
    base_pose[14] = [0.3, 0.0, 0.4]   # Right elbow
    base_pose[15] = [-0.4, 0.0, 0.2]  # Left wrist
    base_pose[16] = [0.4, 0.0, 0.2]   # Right wrist
    base_pose[23] = [-0.1, 0.0, 0.0]  # Left hip
    base_pose[24] = [0.1, 0.0, 0.0]   # Right hip
    
    # Add temporal motion based on gesture type
    sequence = []
    
    for t in range(num_frames):
        frame = base_pose.copy()
        
        # Add gesture-specific motion patterns
        frequency = 0.05 + (gesture_type % 8) * 0.01
        amplitude = 0.15 + (gesture_type % 8) * 0.02
        phase = 2 * np.pi * frequency * t
        
        if gesture_type < 7:  # Emotions (more static)
            # Subtle body movements for emotions
            frame[:, 0] += amplitude * 0.3 * np.sin(phase)  # Slight sway
            frame[:, 1] += amplitude * 0.2 * np.cos(phase)
            
        else:  # Gestures (more dynamic)
            # Dynamic arm movements for gestures
            # Right arm motion
            frame[14:17, 0] += amplitude * np.sin(phase)     # x movement
            frame[14:17, 1] += amplitude * np.cos(phase)     # y movement
            
            # Left arm motion (opposite phase for some gestures)
            if gesture_type % 2 == 0:
                frame[13:16, 0] += amplitude * np.sin(phase + np.pi)
                frame[13:16, 1] += amplitude * np.cos(phase + np.pi)
        
        # Add small random noise for realism
        noise = np.random.randn(num_joints, 3) * 0.01
        frame += noise
        
        # Set high confidence values
        frame[:, 2] = np.clip(np.random.uniform(0.85, 0.98, num_joints), 0, 1)
        
        sequence.append(frame)
    
    return np.array(sequence)


def generate_dataset(
    output_dir: str = 'data/processed',
    num_classes: int = 15,
    samples_per_class: int = 100,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
):
    """
    Generate complete synthetic dataset.
    
    Args:
        output_dir: Where to save the data
        num_classes: Number of classes (15 = 7 emotions + 8 gestures)
        samples_per_class: How many samples per class
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    (output_path / 'train').mkdir(exist_ok=True)
    (output_path / 'val').mkdir(exist_ok=True)
    (output_path / 'test').mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("🎬 SYNTHETIC DATASET GENERATION")
    print("="*60)
    print(f"\n📊 Configuration:")
    print(f"   Classes: {num_classes}")
    print(f"   Samples per class: {samples_per_class}")
    print(f"   Total samples: {num_classes * samples_per_class}")
    print(f"   Train/Val/Test split: {train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")
    
    # Class names for reference
    class_names = [
        'Happy', 'Sad', 'Angry', 'Confused', 'Neutral', 'Excited', 'Fearful',  # Emotions
        'Waving', 'Pointing', 'Asking', 'Signaling', 'Warning', 'Greeting', 'Dismissing', 'Celebrating'  # Gestures
    ]
    
    print(f"\n📝 Classes:")
    for i, name in enumerate(class_names[:num_classes]):
        print(f"   {i:2d}. {name}")
    
    print(f"\n🔨 Generating samples...\n")
    
    total_samples = 0
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    
    for class_id in range(num_classes):
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        print(f"📦 Generating class {class_id:2d} ({class_name})...")
        
        for sample_id in tqdm(range(samples_per_class), desc=f"  Class {class_id:02d}"):
            # Generate sequence
            sequence = generate_sample_sequence(
                num_frames=150,  # Reduced for CPU efficiency
                num_joints=25,
                gesture_type=class_id
            )
            
            # Determine split
            rand = np.random.rand()
            if rand < train_ratio:
                split = 'train'
            elif rand < train_ratio + val_ratio:
                split = 'val'
            else:
                split = 'test'
            
            split_counts[split] += 1
            
            # Save with format: classID_sampleID.npy
            filename = f"{class_id:02d}_{sample_id:04d}.npy"
            filepath = output_path / split / filename
            np.save(filepath, sequence)
            
            total_samples += 1
    
    # Print statistics
    print("\n" + "="*60)
    print("✅ DATASET GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📊 Statistics:")
    print(f"   Total samples generated: {total_samples}")
    print(f"   Training samples: {split_counts['train']}")
    print(f"   Validation samples: {split_counts['val']}")
    print(f"   Test samples: {split_counts['test']}")
    
    # Calculate storage
    sample_file = output_path / 'train' / os.listdir(output_path / 'train')[0]
    file_size = os.path.getsize(sample_file) / 1024  # KB
    total_size = (file_size * total_samples) / 1024  # MB
    
    print(f"\n💾 Storage:")
    print(f"   Size per sample: {file_size:.1f} KB")
    print(f"   Total dataset size: {total_size:.1f} MB")
    
    print(f"\n📂 Location:")
    print(f"   {output_path.absolute()}")
    
    print("\n" + "="*60)
    print("✅ Ready for training!")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Synthetic Skeleton Dataset')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--classes', type=int, default=15,
                       help='Number of classes')
    parser.add_argument('--samples', type=int, default=100,
                       help='Samples per class')
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        num_classes=args.classes,
        samples_per_class=args.samples
    )
    
    print("🎉 Dataset generation complete! You can now train the model.\n")