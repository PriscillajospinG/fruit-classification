import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
from src.fruit_detector import FruitClassifier

class FruitTrainer:
    def __init__(self, data_dir):
        """Initialize trainer with data directory"""
        self.data_dir = data_dir
        self.classifier = FruitClassifier()
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
    
    def load_data_from_directory(self):
        """Load training data from directory structure"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classifier.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found. Skipping {class_name}")
                continue
            
            print(f"Loading images for {class_name}...")
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.classifier.img_size)
                            img = img.astype('float32') / 255.0
                            
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        if len(images) == 0:
            raise ValueError("No images found. Please check your data directory structure.")
        
        print(f"Loaded {len(images)} images total")
        return np.array(images), np.array(labels)
    
    def create_synthetic_data(self, samples_per_class=100):
        """Create synthetic data for demonstration purposes"""
        print("Creating synthetic training data...")
        
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classifier.class_names):
            print(f"Generating {samples_per_class} samples for {class_name}")
            
            for _ in range(samples_per_class):
                # Create a synthetic image with some patterns
                img = np.random.rand(224, 224, 3).astype('float32')
                
                # Add some class-specific patterns
                if class_name == 'apple':
                    img[:, :, 0] += 0.3  # More red
                elif class_name == 'banana':
                    img[:, :, 1] += 0.3  # More yellow
                    img[:, :, 0] += 0.3
                elif class_name == 'orange':
                    img[:, :, 0] += 0.4  # Orange color
                    img[:, :, 1] += 0.2
                elif class_name == 'grape':
                    img[:, :, 2] += 0.3  # More purple/blue
                elif class_name == 'strawberry':
                    img[:, :, 0] += 0.4  # Red
                    img[100:124, 100:124, 1] += 0.3  # Green top
                
                # Normalize
                img = np.clip(img, 0, 1)
                
                images.append(img)
                labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, use_synthetic=True):
        """Prepare training and validation data"""
        try:
            if use_synthetic:
                X, y = self.create_synthetic_data()
            else:
                X, y = self.load_data_from_directory()
            
            # Split data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training samples: {len(self.X_train)}")
            print(f"Validation samples: {len(self.X_val)}")
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise
    
    def train_model(self, epochs=20, batch_size=32):
        """Train the fruit classification model"""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        print("Starting model training...")
        
        # Data augmentation
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ])
        
        # Update model with data augmentation
        inputs = keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(2, 2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        outputs = keras.layers.Dense(len(self.classifier.class_names), activation='softmax')(x)
        
        self.classifier.model = keras.Model(inputs, outputs)
        
        self.classifier.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Train model
        history = self.classifier.model.fit(
            self.X_train, self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.X_val is None:
            raise ValueError("Validation data not available")
        
        loss, accuracy = self.classifier.model.evaluate(self.X_val, self.y_val, verbose=0)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

if __name__ == "__main__":
    # Example usage
    trainer = FruitTrainer("data/fruits")  # Directory with fruit images
    
    # Prepare data (using synthetic data for demo)
    trainer.prepare_data(use_synthetic=True)
    
    # Train model
    history = trainer.train_model(epochs=10)
    
    # Evaluate
    trainer.evaluate_model()
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Save trained model
    trainer.classifier.save_model("models/fruit_classifier.h5")
