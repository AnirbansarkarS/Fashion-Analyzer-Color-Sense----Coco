"""Dataset Preprocessing Script"""

import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

class FashionDatasetPreprocessor:
    """Preprocess and prepare fashion datasets"""
    
    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def prepare_custom_dataset(self, raw_data_dir, train_split=0.8):
        """Prepare custom fashion dataset"""
        images = []
        scores = []
        
        print("📂 Loading images...")
        for score_folder in sorted(os.listdir(raw_data_dir)):
            score_path = os.path.join(raw_data_dir, score_folder)
            
            if not os.path.isdir(score_path):
                continue
            
            try:
                score = int(score_folder.split('_')[1]) / 10.0
            except:
                continue
            
            for img_file in os.listdir(score_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(score_path, img_file)
                    
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        img = cv2.resize(img, (224, 224))
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        scores.append(score)
                        
                    except Exception as e:
                        print(f"⚠ Error processing {img_file}: {e}")
                        continue
        
        print(f"✓ Loaded {len(images)} images")
        
        if len(images) == 0:
            print("✗ No images found!")
            return None
        
        X = np.array(images)
        y = np.array(scores).reshape(-1, 1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-train_split, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"✓ Train set: {X_train.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        print(f"✓ Test set: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def save_dataset(self, dataset_dict, name='fashion_dataset'):
        """Save preprocessed dataset"""
        save_path = os.path.join(self.output_dir, f'{name}.pkl')
        
        with open(save_path, 'wb') as f:
            pickle.dump(dataset_dict, f)
        
        print(f"✓ Dataset saved to {save_path}")
    
    def load_dataset(self, name='fashion_dataset'):
        """Load preprocessed dataset"""
        load_path = os.path.join(self.output_dir, f'{name}.pkl')
        
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        
        print(f"✓ Dataset loaded from {load_path}")
        return dataset
    
    def prepare_fashion_mnist(self):
        """Download and prepare Fashion-MNIST dataset"""
        from tensorflow.keras.datasets import fashion_mnist
        from tensorflow.image import resize
        
        print("📥 Downloading Fashion-MNIST...")
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        X_train = np.repeat(np.expand_dims(X_train, axis=-1), 3, axis=-1)
        X_test = np.repeat(np.expand_dims(X_test, axis=-1), 3, axis=-1)
        
        print("🔄 Resizing images...")
        X_train = resize(X_train, (224, 224)).numpy()
        X_test = resize(X_test, (224, 224)).numpy()
        
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        y_train = (y_train / 10.0).astype('float32').reshape(-1, 1)
        y_test = (y_test / 10.0).astype('float32').reshape(-1, 1)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'dataset_name': 'Fashion-MNIST'
        }
        
        print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return dataset