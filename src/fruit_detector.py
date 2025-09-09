import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os

class FruitClassifier:
    def __init__(self, model_path=None):
        """Initialize the fruit classifier with a pre-trained model or create a new one"""
        self.class_names = ['apple', 'banana', 'orange', 'grape', 'strawberry', 'kiwi', 'mango', 'pineapple']
        self.img_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
        else:
            self.model = self.create_model()
            print("Created new model. Train it before using for predictions.")
    
    def create_model(self):
        """Create a CNN model for fruit classification"""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        image_resized = cv2.resize(image, self.img_size)
        # Normalize pixel values
        image_normalized = image_resized.astype('float32') / 255.0
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        return image_batch
    
    def predict(self, image):
        """Predict the fruit in the image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return self.class_names[predicted_class], confidence
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")

class FruitDetector:
    def __init__(self, model_path=None):
        """Initialize the fruit detector with camera and classifier"""
        self.classifier = FruitClassifier(model_path)
        self.camera = cv2.VideoCapture(0)
        
        # Check if camera is working
        if not self.camera.isOpened():
            raise ValueError("Could not open camera")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
    
    def detect_fruits_live(self):
        """Start live fruit detection from camera"""
        print("Starting live fruit detection. Press 'q' to quit.")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create a region of interest (ROI) for detection
            height, width = frame.shape[:2]
            roi_size = 300
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2
            
            # Extract ROI
            roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            
            # Predict fruit in ROI
            try:
                fruit_name, confidence = self.classifier.predict(roi)
                
                # Display prediction if confidence is high enough
                if confidence > 0.5:
                    text = f"{fruit_name}: {confidence:.2f}"
                    color = (0, 255, 0)  # Green for high confidence
                else:
                    text = "Unknown fruit"
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw ROI rectangle
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), color, 2)
                
                # Display prediction text
                cv2.putText(frame, text, (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Display instructions
                cv2.putText(frame, "Place fruit in the green box", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Error in prediction", (roi_x, roi_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Fruit Detection', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
    
    def capture_and_predict(self):
        """Capture a single frame and predict"""
        ret, frame = self.camera.read()
        if ret:
            fruit_name, confidence = self.classifier.predict(frame)
            return fruit_name, confidence, frame
        return None, None, None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize detector (you can provide a model path if you have a trained model)
        detector = FruitDetector()
        
        # Start live detection
        detector.detect_fruits_live()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible.")
