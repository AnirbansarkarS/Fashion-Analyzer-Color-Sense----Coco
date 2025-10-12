"""Quick Setup and Training Script"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    dirs = [
        'models',
        'data/raw',
        'data/processed',
        'evaluation',
        'logs',
        'model_training',
        'utils'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    os.system(f"{sys.executable} -m pip install -r requirements.txt")

def prepare_dataset():
    """Download and prepare dataset"""
    print("\n📥 Preparing dataset...")
    try:
        from model_training.dataset_preprocess import FashionDatasetPreprocessor
        
        processor = FashionDatasetPreprocessor()
        dataset = processor.prepare_fashion_mnist()
        processor.save_dataset(dataset, 'fashion_mnist')
        print("✓ Dataset prepared successfully!")
        return dataset
    except Exception as e:
        print(f"✗ Error preparing dataset: {e}")
        return None

def train_model(dataset=None):
    """Train the model"""
    print("\n🚀 Training model...")
    try:
        from model_training.train_model import FashionModel
        import numpy as np
        
        if dataset is None:
            print("⚠ No dataset provided. Creating synthetic data...")
            dataset = {
                'X_train': np.random.rand(100, 224, 224, 3).astype('float32'),
                'y_train': np.random.rand(100, 1).astype('float32'),
                'X_val': np.random.rand(20, 224, 224, 3).astype('float32'),
                'y_val': np.random.rand(20, 1).astype('float32'),
            }
        
        model = FashionModel()
        model.build_model()
        
        print("\nModel Summary:")
        model.model.summary()
        
        print("\nStarting training...")
        history = model.train(
            X_train=dataset['X_train'],
            y_train=dataset['y_train'],
            X_val=dataset['X_val'],
            y_val=dataset['y_val'],
            epochs=20,
            batch_size=32
        )
        
        model.save_model('models/fashion_model.h5')
        print("✓ Model trained and saved successfully!")
        return model, history
    
    except Exception as e:
        print(f"✗ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main setup flow"""
    print("="*60)
    print("🎨 FASHION ANALYZER - SETUP & TRAINING")
    print("="*60)
    
    # Step 1: Create directories
    print("\n[1/4] Creating directories...")
    create_directories()
    
    # Step 2: Install dependencies
    print("\n[2/4] Installing dependencies...")
    response = input("Install dependencies? (y/n): ").lower()
    if response == 'y':
        install_dependencies()
    else:
        print("⚠ Skipped dependency installation")
    
    # Step 3: Prepare dataset
    print("\n[3/4] Preparing dataset...")
    response = input("Download and prepare Fashion-MNIST dataset? (y/n): ").lower()
    dataset = None
    if response == 'y':
        dataset = prepare_dataset()
    
    # Step 4: Train model
    print("\n[4/4] Training model...")
    response = input("Train model? (y/n): ").lower()
    if response == 'y':
        train_model(dataset)
    
    # Final info
    print("\n" + "="*60)
    print("✓ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the API: python main.py")
    print("2. API will be available at: http://localhost:8000")
    print("3. Visit http://localhost:8000/docs for interactive API docs")
    print("4. Upload images to /analyze endpoint for fashion scoring")
    print("="*60)

if __name__ == "__main__":
    main()