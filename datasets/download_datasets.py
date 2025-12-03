"""
Automated Dataset Download Script

Downloads and prepares datasets for training:
1. NTU RGB+D Skeleton Dataset
2. OpenPose Gesture Datasets from Kaggle
3. Custom Body Language Datasets

Requirements:
- Kaggle API configured (kaggle.json in ~/.kaggle/)
- Sufficient disk space (10+ GB)
"""

import os
import sys
import argparse
import zipfile
import gdown
from pathlib import Path
from tqdm import tqdm
import shutil


class DatasetDownloader:
    """Download and prepare datasets for training."""
    
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'ntu_rgbd': {
                'name': 'NTU RGB+D Skeleton',
                'size': '5.8 GB',
                'url': 'https://drive.google.com/uc?id=1CUZnBtYwifVXS21yVg62T-vrPVayso5H',
                'type': 'google_drive'
            },
            'openpose_gestures': {
                'name': 'OpenPose Gesture Dataset',
                'kaggle_dataset': 'meetnagadia/openpose-dataset',
                'type': 'kaggle'
            },
            'body_language': {
                'name': 'Body Language Dataset',
                'kaggle_dataset': 'shawngustaw/body-language-dataset',
                'type': 'kaggle'
            },
            'emotion_poses': {
                'name': 'Emotion from Poses Dataset',
                'kaggle_dataset': 'gauravsharma99/ucf101-action-recognition-video-classification',
                'type': 'kaggle'
            }
        }
    
    def check_kaggle_setup(self):
        """Check if Kaggle API is properly configured."""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("\n❌ Kaggle API not configured!")
            print("\nSetup instructions:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. Place kaggle.json in ~/.kaggle/")
            print("\nCommands:")
            print("  mkdir -p ~/.kaggle")
            print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
            print("  chmod 600 ~/.kaggle/kaggle.json")
            return False
        
        # Check permissions
        if kaggle_json.stat().st_mode & 0o777 != 0o600:
            print("⚠️  Fixing kaggle.json permissions...")
            os.chmod(kaggle_json, 0o600)
        
        print("✅ Kaggle API configured")
        return True
    
    def download_from_kaggle(self, dataset_name):
        """Download dataset from Kaggle."""
        try:
            import kaggle
        except ImportError:
            print("❌ Kaggle package not installed!")
            print("Install with: pip install kaggle")
            return False
        
        dataset_info = self.datasets[dataset_name]
        kaggle_dataset = dataset_info['kaggle_dataset']
        
        output_path = self.output_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📥 Downloading {dataset_info['name']} from Kaggle...")
        print(f"Dataset: {kaggle_dataset}")
        print(f"Output: {output_path}")
        
        try:
            # Download dataset
            kaggle.api.dataset_download_files(
                kaggle_dataset,
                path=output_path,
                unzip=True,
                quiet=False
            )
            
            print(f"✅ Downloaded {dataset_info['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def download_from_google_drive(self, dataset_name):
        """Download dataset from Google Drive."""
        dataset_info = self.datasets[dataset_name]
        
        output_path = self.output_dir / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        zip_path = output_path / f"{dataset_name}.zip"
        
        print(f"\n📥 Downloading {dataset_info['name']} from Google Drive...")
        print(f"Size: {dataset_info['size']}")
        print(f"URL: {dataset_info['url']}")
        
        try:
            # Download file
            gdown.download(dataset_info['url'], str(zip_path), quiet=False)
            
            # Extract zip
            print(f"\n📦 Extracting {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in tqdm(zip_ref.infolist(), desc='Extracting'):
                    zip_ref.extract(member, output_path)
            
            # Remove zip file
            zip_path.unlink()
            
            print(f"✅ Downloaded and extracted {dataset_info['name']}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_name}: {e}")
            return False
    
    def download_dataset(self, dataset_name):
        """Download specific dataset."""
        if dataset_name not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_name}")
            print(f"Available datasets: {', '.join(self.datasets.keys())}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        
        # Check if already downloaded
        output_path = self.output_dir / dataset_name
        if output_path.exists() and any(output_path.iterdir()):
            print(f"\n⚠️  Dataset {dataset_name} already exists at {output_path}")
            response = input("Re-download? (y/n): ")
            if response.lower() != 'y':
                return True
            shutil.rmtree(output_path)
        
        # Download based on type
        if dataset_info['type'] == 'kaggle':
            return self.download_from_kaggle(dataset_name)
        elif dataset_info['type'] == 'google_drive':
            return self.download_from_google_drive(dataset_name)
        else:
            print(f"❌ Unknown dataset type: {dataset_info['type']}")
            return False
    
    def download_all(self):
        """Download all available datasets."""
        print("\n" + "="*60)
        print("DOWNLOADING ALL DATASETS")
        print("="*60)
        
        success_count = 0
        total_count = len(self.datasets)
        
        for dataset_name in self.datasets:
            if self.download_dataset(dataset_name):
                success_count += 1
            print()
        
        print("="*60)
        print(f"DOWNLOAD SUMMARY: {success_count}/{total_count} successful")
        print("="*60)
    
    def list_datasets(self):
        """List all available datasets."""
        print("\n" + "="*60)
        print("AVAILABLE DATASETS")
        print("="*60)
        
        for name, info in self.datasets.items():
            print(f"\n📊 {name}")
            print(f"   Name: {info['name']}")
            if 'size' in info:
                print(f"   Size: {info['size']}")
            if 'kaggle_dataset' in info:
                print(f"   Kaggle: {info['kaggle_dataset']}")
            if 'url' in info:
                print(f"   URL: {info['url'][:50]}...")
        
        print("\n" + "="*60)
    
    def verify_downloads(self):
        """Verify all downloaded datasets."""
        print("\n" + "="*60)
        print("VERIFYING DOWNLOADS")
        print("="*60)
        
        for dataset_name in self.datasets:
            dataset_path = self.output_dir / dataset_name
            
            if dataset_path.exists():
                file_count = sum(1 for _ in dataset_path.rglob('*') if _.is_file())
                size_mb = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file()) / (1024 * 1024)
                
                print(f"\n✅ {dataset_name}")
                print(f"   Path: {dataset_path}")
                print(f"   Files: {file_count:,}")
                print(f"   Size: {size_mb:.1f} MB")
            else:
                print(f"\n❌ {dataset_name} - Not found")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Download datasets for Pose-based Gesture Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific dataset
  python download_datasets.py --dataset ntu_rgbd
  
  # Download all datasets
  python download_datasets.py --all
  
  # List available datasets
  python download_datasets.py --list
  
  # Verify downloaded datasets
  python download_datasets.py --verify
        """
    )
    
    parser.add_argument('--dataset', type=str, 
                       help='Specific dataset to download')
    parser.add_argument('--all', action='store_true',
                       help='Download all datasets')
    parser.add_argument('--list', action='store_true',
                       help='List available datasets')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded datasets')
    parser.add_argument('--output', type=str, default='data/raw',
                       help='Output directory (default: data/raw)')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = DatasetDownloader(output_dir=args.output)
    
    # Handle commands
    if args.list:
        downloader.list_datasets()
    
    elif args.verify:
        downloader.verify_downloads()
    
    elif args.all:
        # Check Kaggle setup
        if not downloader.check_kaggle_setup():
            print("\n⚠️  Kaggle datasets will be skipped")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
        
        downloader.download_all()
    
    elif args.dataset:
        # Check Kaggle setup if needed
        dataset_info = downloader.datasets.get(args.dataset)
        if dataset_info and dataset_info['type'] == 'kaggle':
            if not downloader.check_kaggle_setup():
                print("\n❌ Cannot download Kaggle dataset without API setup")
                return
        
        downloader.download_dataset(args.dataset)
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        downloader.list_datasets()


if __name__ == '__main__':
    main()