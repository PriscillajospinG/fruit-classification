import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import matplotlib.pyplot as plt
from collections import Counter

class FruitsDatasetTrainer:
    def __init__(self, dataset_path="data/fruits-360_100x100/fruits-360"):
        """Initialize trainer with Fruits-360 dataset"""
        self.dataset_path = dataset_path
        self.train_dir = os.path.join(dataset_path, "Training")
        self.test_dir = os.path.join(dataset_path, "Test")
        self.img_size = (100, 100)  # Fruits-360 images are 100x100
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
        # Common fruit categories we want to focus on
        self.target_fruits = [
            'Apple', 'Banana', 'Orange', 'Grape', 'Strawberry', 
            'Kiwi', 'Mango', 'Pineapple', 'Pear', 'Cherry',
            'Lemon', 'Peach', 'Plum', 'Avocado'
        ]
    
    def get_fruit_categories(self):
        """Get all available fruit categories from the dataset"""
        if not os.path.exists(self.train_dir):
            raise ValueError(f"Training directory not found: {self.train_dir}")
        
        all_categories = [d for d in os.listdir(self.train_dir) 
                         if os.path.isdir(os.path.join(self.train_dir, d))]
        
        # Filter categories that match our target fruits
        filtered_categories = []
        for category in all_categories:
            for target in self.target_fruits:
                if target.lower() in category.lower():
                    filtered_categories.append(category)
                    break
        
        print(f"Found {len(all_categories)} total categories")
        print(f"Using {len(filtered_categories)} fruit categories")
        
        return filtered_categories
    
    def load_images_from_category(self, category_path, max_images_per_category=200):
        """Load images from a specific category directory"""
        images = []
        image_files = [f for f in os.listdir(category_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit images per category to prevent memory issues
        image_files = image_files[:max_images_per_category]
        
        for img_file in image_files:
            img_path = os.path.join(category_path, img_file)
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Resize to target size (should already be 100x100 for Fruits-360)
                    img = cv2.resize(img, self.img_size)
                    # Normalize
                    img = img.astype('float32') / 255.0
                    images.append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        return images
    
    def load_dataset(self, max_images_per_category=200):
        """Load the complete dataset"""
        print("Loading Fruits-360 dataset...")
        
        categories = self.get_fruit_categories()
        
        all_images = []
        all_labels = []
        
        for category in categories:
            category_path = os.path.join(self.train_dir, category)
            
            print(f"Loading category: {category}")
            images = self.load_images_from_category(category_path, max_images_per_category)
            
            if len(images) > 0:
                all_images.extend(images)
                all_labels.extend([category] * len(images))
                print(f"  Loaded {len(images)} images")
            else:
                print(f"  No images found in {category}")
        
        if len(all_images) == 0:
            raise ValueError("No images loaded from dataset")
        
        # Convert to numpy arrays
        X = np.array(all_images)
        
        # Encode labels
        y = self.label_encoder.fit_transform(all_labels)
        self.class_names = self.label_encoder.classes_
        
        print(f"\nDataset loaded successfully:")
        print(f"Total images: {len(X)}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Image shape: {X.shape[1:]}")
        
        # Show class distribution
        label_counts = Counter(all_labels)
        print("\nClass distribution:")
        for class_name, count in label_counts.most_common():
            print(f"  {class_name}: {count} images")
        
        return X, y
    
    def create_model(self, num_classes):
        """Create CNN model for fruit classification"""
        model = keras.Sequential([
            # Input layer
            keras.layers.Input(shape=(100, 100, 3)),
            
            # Data augmentation
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            
            # Convolutional layers
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2, 2),
            
            # Classifier
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X, y, epochs=30, batch_size=32, validation_split=0.2):
        """Train the fruit classification model"""
        print("Preparing model for training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create model
        num_classes = len(self.class_names)
        self.model = self.create_model(num_classes)
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.2, 
                patience=5,
                monitor='val_accuracy'
            ),
            keras.callbacks.ModelCheckpoint(
                'models/best_fruits_model.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X, y):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nFinal Model Performance:")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Get predictions for confusion matrix
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        
        # Print classification report
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return loss, accuracy
    
    def save_model(self, filepath="models/fruits360_classifier.h5"):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        
        # Save class names
        import pickle
        with open(filepath.replace('.h5', '_classes.pkl'), 'wb') as f:
            pickle.dump(self.class_names, f)
        
        print(f"Model saved to {filepath}")
        print(f"Class names saved to {filepath.replace('.h5', '_classes.pkl')}")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('fruits360_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    try:
        # Initialize trainer
        trainer = FruitsDatasetTrainer()
        
        # Load dataset
        X, y = trainer.load_dataset(max_images_per_category=150)
        
        # Train model
        history = trainer.train_model(X, y, epochs=25, batch_size=32)
        
        # Evaluate model
        trainer.evaluate_model(X, y)
        
        # Save model
        trainer.save_model()
        
        # Plot training history
        trainer.plot_training_history(history)
        
        print("\n🎉 Training completed successfully!")
        print("You can now use the trained model for live detection:")
        print("python main.py live")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
