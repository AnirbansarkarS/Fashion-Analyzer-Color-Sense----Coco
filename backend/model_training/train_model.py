import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path

class FashionModel:
    """
    Fashion scoring model using CNN
    Can be fine-tuned with custom dataset
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self):
        """
        Build CNN model with transfer learning (MobileNetV2)
        """
        # Load pre-trained MobileNetV2
        base_model = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Output: 0-1 (normalized score)
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def save_model(self, model_path):
        """Save model"""
        if self.model is None:
            print("✗ Model not built yet")
            return
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None, 
              epochs=20, batch_size=32):
        """
        Train the model
        If no data provided, uses synthetic data for demo
        """
        
        # If no training data provided, create synthetic data
        if X_train is None:
            print("⚠ No training data provided. Using synthetic data for demo...")
            X_train = np.random.rand(100, 224, 224, 3).astype('float32')
            y_train = np.random.rand(100, 1).astype('float32')
            X_val = np.random.rand(20, 224, 224, 3).astype('float32')
            y_val = np.random.rand(20, 1).astype('float32')
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train
        print("\n🚀 Starting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training complete!")
        return self.history
    
    def predict(self, image, return_features=False):
        """
        Predict fashion score for image
        """
        if self.model is None:
            print("✗ Model not loaded")
            return None
        
        try:
            # Ensure correct shape
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Predict
            prediction = self.model.predict(image, verbose=0)[0][0]
            score = prediction * 10  # Scale from 0-10
            
            if return_features:
                # Extract intermediate features
                feature_extractor = models.Model(
                    inputs=self.model.input,
                    outputs=self.model.layers[-3].output
                )
                features = feature_extractor.predict(image, verbose=0)
                
                return {
                    'score': score,
                    'features': features.flatten()
                }
            
            return {
                'style_match': score / 10,
                'texture_quality': (score / 10) * 0.9,
                'confidence': float(prediction)
            }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.model is None:
            print("✗ Model not loaded")
            return None
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        return {'loss': loss, 'mae': mae}
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=10):
        """
        Fine-tune model with custom dataset
        Unfreezes some base model layers
        """
        if self.model is None:
            print("✗ Model not built yet")
            return
        
        # Unfreeze last 50 layers of base model
        for layer in self.model.layers[0].layers[-50:]:
            layer.trainable = True
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        print("🔧 Fine-tuning model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        print("✓ Fine-tuning complete!")
        return self.history

        import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path

class FashionModel:
    """Fashion scoring model using CNN with transfer learning"""
    
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build CNN model with transfer learning (MobileNetV2)"""
        try:
            # Load pre-trained MobileNetV2
            base_model = keras.applications.MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom top layers
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='sigmoid')  # Output: 0-1
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.model = model
            print("✓ Model built successfully")
            return model
        except Exception as e:
            print(f"✗ Error building model: {e}")
            raise
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise
    
    def save_model(self, model_path):
        """Save model"""
        if self.model is None:
            print("✗ Model not built yet")
            return
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_path)
        print(f"✓ Model saved to {model_path}")
    
    def train(self, X_train=None, y_train=None, X_val=None, y_val=None, 
              epochs=20, batch_size=32):
        """Train the model"""
        
        # If no training data provided, create synthetic data
        if X_train is None:
            print("⚠ No training data provided. Creating synthetic data for demo...")
            X_train = np.random.rand(100, 224, 224, 3).astype('float32')
            y_train = np.random.rand(100, 1).astype('float32')
            X_val = np.random.rand(20, 224, 224, 3).astype('float32')
            y_val = np.random.rand(20, 1).astype('float32')
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # Train
        print("\n🚀 Starting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✓ Training complete!")
        return self.history
    
    def predict(self, image, return_features=False):
        """Predict fashion score for image"""
        if self.model is None:
            print("✗ Model not loaded")
            return None
        
        try:
            # Ensure correct shape
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Predict
            prediction = self.model.predict(image, verbose=0)[0][0]
            score = prediction * 10  # Scale from 0-10
            
            return {
                'style_match': score / 10,
                'texture_quality': (score / 10) * 0.9,
                'confidence': float(prediction)
            }
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        if self.model is None:
            print("✗ Model not loaded")
            return None
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        return {'loss': loss, 'mae': mae}
    
    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=10):
        """Fine-tune model with custom dataset"""
        if self.model is None:
            print("✗ Model not built yet")
            return
        
        # Unfreeze last 50 layers of base model
        for layer in self.model.layers[0].layers[-50:]:
            layer.trainable = True
        
        # Use lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
        
        print("🔧 Fine-tuning model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            verbose=1
        )
        
        print("✓ Fine-tuning complete!")
        return self.history

class DataGenerator:
    """Generate training data from fashion datasets"""
    
    @staticmethod
    def load_fashion_mnist():
        """Load Fashion-MNIST dataset for quick prototyping"""
        from tensorflow.keras.datasets import fashion_mnist
        from tensorflow.image import resize
        
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
        # Convert to 224x224 for model input
        X_train = np.repeat(np.expand_dims(X_train, axis=-1), 3, axis=-1)
        X_test = np.repeat(np.expand_dims(X_test, axis=-1), 3, axis=-1)
        
        # Resize
        X_train = resize(X_train, (224, 224)).numpy()
        X_test = resize(X_test, (224, 224)).numpy()
        
        # Normalize
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Create regression targets (0-1 scores)
        y_train = (y_train / 10.0).astype('float32').reshape(-1, 1)
        y_test = (y_test / 10.0).astype('float32').reshape(-1, 1)
        
        return (X_train, y_train), (X_test, y_test)