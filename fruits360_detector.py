import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import pickle

class Fruits360Classifier:
    def __init__(self, model_path=None, classes_path=None):
        """Initialize the fruit classifier with Fruits-360 trained model"""
        
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
            
            # Load class names
            if classes_path and os.path.exists(classes_path):
                with open(classes_path, 'rb') as f:
                    self.class_names = pickle.load(f)
            else:
                # Try to find classes file automatically
                auto_classes_path = model_path.replace('.h5', '_classes.pkl')
                if os.path.exists(auto_classes_path):
                    with open(auto_classes_path, 'rb') as f:
                        self.class_names = pickle.load(f)
                else:
                    # Fallback to common fruit names
                    self.class_names = ['Apple', 'Banana', 'Orange', 'Grape', 'Strawberry', 
                                      'Kiwi', 'Mango', 'Pineapple', 'Pear', 'Cherry']
            
            print(f"Loaded {len(self.class_names)} fruit classes")
            
        else:
            # Create a basic model if no trained model available
            self.model = self.create_basic_model()
            self.class_names = ['Apple', 'Banana', 'Orange', 'Grape', 'Strawberry', 
                              'Kiwi', 'Mango', 'Pineapple', 'Pear', 'Cherry']
            print("No trained model found. Using basic model (predictions will be random).")
        
        self.img_size = (100, 100)  # Fruits-360 uses 100x100 images
    
    def create_basic_model(self):
        """Create a basic model structure"""
        model = keras.Sequential([
            keras.layers.Input(shape=(100, 100, 3)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')  # Default 10 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image to 100x100 (Fruits-360 format)
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
        
        # Make sure we don't go out of bounds
        if predicted_class < len(self.class_names):
            fruit_name = self.class_names[predicted_class]
        else:
            fruit_name = "Unknown"
        
        return fruit_name, confidence
    
    def get_top_predictions(self, image, top_k=3):
        """Get top K predictions for an image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)[0]
        
        # Get top K indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(self.class_names):
                fruit_name = self.class_names[idx]
                confidence = predictions[idx]
                results.append((fruit_name, confidence))
        
        return results

class Fruits360Detector:
    def __init__(self, model_path=None, classes_path=None):
        """Initialize the fruit detector with camera and classifier"""
        self.classifier = Fruits360Classifier(model_path, classes_path)
        self.camera = cv2.VideoCapture(0)
        
        # Check if camera is working
        if not self.camera.isOpened():
            raise ValueError("Could not open camera")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
    
    def detect_fruits_live(self):
        """Start live fruit detection from camera"""
        print("Starting live fruit detection with Fruits-360 model...")
        print("Controls:")
        print("  - Place fruit in the green rectangle")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'p' to show top 3 predictions")
        
        show_top_predictions = False
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Create region of interest (ROI)
            height, width = frame.shape[:2]
            roi_size = 250  # Adjusted for better visibility
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2
            
            # Extract ROI
            roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
            
            # Predict fruit in ROI
            try:
                if show_top_predictions:
                    # Show top 3 predictions
                    top_predictions = self.classifier.get_top_predictions(roi, top_k=3)
                    
                    y_offset = roi_y - 100
                    for i, (fruit_name, confidence) in enumerate(top_predictions):
                        if confidence > 0.1:  # Only show if some confidence
                            text = f"{i+1}. {fruit_name}: {confidence:.2f}"
                            color = (0, 255, 0) if i == 0 else (0, 255, 255)
                            cv2.putText(frame, text, (roi_x, y_offset + i*25), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw ROI rectangle
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 255), 2)
                else:
                    # Show single best prediction
                    fruit_name, confidence = self.classifier.predict(roi)
                    
                    # Display prediction if confidence is reasonable
                    if confidence > 0.1:  # Lower threshold for real data
                        text = f"{fruit_name}: {confidence:.2f}"
                        color = (0, 255, 0) if confidence > 0.3 else (0, 255, 255)
                    else:
                        text = "Place fruit in box"
                        color = (255, 255, 255)
                    
                    # Draw ROI rectangle
                    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), color, 2)
                    
                    # Display prediction text
                    cv2.putText(frame, text, (roi_x, roi_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Display instructions
                cv2.putText(frame, "Controls: 'q'-quit, 's'-save, 'p'-toggle predictions", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Model: {len(self.classifier.class_names)} fruit classes", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Error in prediction", (roi_x, roi_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Fruits-360 Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(f'captured_frame_{np.random.randint(1000, 9999)}.jpg', frame)
                print("Frame saved!")
            elif key == ord('p'):
                # Toggle prediction display mode
                show_top_predictions = not show_top_predictions
                mode = "top 3" if show_top_predictions else "single"
                print(f"Switched to {mode} prediction mode")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
    
    def test_prediction(self, image_path):
        """Test prediction on a saved image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        fruit_name, confidence = self.classifier.predict(image_rgb)
        top_predictions = self.classifier.get_top_predictions(image_rgb, top_k=3)
        
        print(f"\nPrediction for {image_path}:")
        print(f"Best prediction: {fruit_name} ({confidence:.3f})")
        print("Top 3 predictions:")
        for i, (name, conf) in enumerate(top_predictions):
            print(f"  {i+1}. {name}: {conf:.3f}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    try:
        # Check for trained model
        model_path = "models/fruits360_classifier.h5"
        classes_path = "models/fruits360_classifier_classes.pkl"
        
        if not os.path.exists(model_path):
            print("No trained Fruits-360 model found.")
            print("Please run: python train_fruits360.py")
            print("Using basic model for demonstration...")
            model_path = None
        
        # Initialize detector
        detector = Fruits360Detector(model_path, classes_path)
        
        # Start live detection
        detector.detect_fruits_live()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible.")
