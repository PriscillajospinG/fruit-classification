"""
Data collection script to help gather fruit images for training
"""

import cv2
import os
import time

class DataCollector:
    def __init__(self, data_dir="data/fruits"):
        """Initialize data collector"""
        self.data_dir = data_dir
        self.camera = cv2.VideoCapture(0)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Fruit classes
        self.fruit_classes = ['apple', 'banana', 'orange', 'grape', 'strawberry', 'kiwi', 'mango', 'pineapple']
        
        # Create directories for each fruit class
        for fruit in self.fruit_classes:
            os.makedirs(os.path.join(data_dir, fruit), exist_ok=True)
    
    def collect_images(self, fruit_name, num_images=50):
        """Collect images for a specific fruit"""
        if fruit_name not in self.fruit_classes:
            print(f"Unknown fruit: {fruit_name}")
            print(f"Available fruits: {', '.join(self.fruit_classes)}")
            return
        
        save_dir = os.path.join(self.data_dir, fruit_name)
        
        print(f"Collecting {num_images} images for {fruit_name}")
        print("Position the fruit in front of the camera")
        print("Press SPACE to capture image, 'q' to quit")
        
        captured = 0
        
        while captured < num_images:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add text overlay
            cv2.putText(frame, f"Collecting: {fruit_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {captured}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, 'q' to quit", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw capture area
            height, width = frame.shape[:2]
            roi_size = 300
            roi_x = (width - roi_size) // 2
            roi_y = (height - roi_size) // 2
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 2)
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                # Extract ROI
                roi = frame[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
                
                # Save image
                timestamp = int(time.time() * 1000)
                filename = f"{fruit_name}_{timestamp}_{captured+1}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                cv2.imwrite(filepath, roi)
                captured += 1
                print(f"Captured {captured}/{num_images}: {filename}")
                
                # Small delay to avoid accidental multiple captures
                time.sleep(0.5)
            
            elif key == ord('q'):  # Quit
                break
        
        print(f"Collection complete! Captured {captured} images for {fruit_name}")
    
    def interactive_collection(self):
        """Interactive data collection for all fruits"""
        print("Interactive Data Collection")
        print("=" * 30)
        
        for fruit in self.fruit_classes:
            print(f"\nNext fruit: {fruit}")
            response = input(f"Collect images for {fruit}? (y/n/skip): ").lower()
            
            if response == 'y':
                num_images = input("Number of images to collect (default 20): ")
                try:
                    num_images = int(num_images) if num_images else 20
                except ValueError:
                    num_images = 20
                
                self.collect_images(fruit, num_images)
            elif response == 'skip':
                continue
            else:
                break
        
        print("\nData collection session completed!")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'camera'):
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DataCollector()
    
    print("Fruit Data Collection Tool")
    print("=" * 30)
    print("1. Interactive collection (all fruits)")
    print("2. Single fruit collection")
    print("3. Exit")
    
    choice = input("Choose option (1-3): ")
    
    if choice == '1':
        collector.interactive_collection()
    elif choice == '2':
        print(f"Available fruits: {', '.join(collector.fruit_classes)}")
        fruit = input("Enter fruit name: ").lower()
        num_images = input("Number of images (default 20): ")
        try:
            num_images = int(num_images) if num_images else 20
        except ValueError:
            num_images = 20
        
        collector.collect_images(fruit, num_images)
    else:
        print("Exiting...")
