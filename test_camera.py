#!/usr/bin/env python3
"""
Test script to check if camera is working properly
"""

import cv2
import sys

def test_camera():
    """Test if camera is accessible and working"""
    print("Testing camera access...")
    
    try:
        # Try to open camera
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            print("❌ Error: Could not open camera")
            print("Troubleshooting:")
            print("1. Make sure your camera is connected")
            print("2. Check if another application is using the camera")
            print("3. Try a different camera index (1, 2, etc.)")
            return False
        
        print("✅ Camera opened successfully")
        
        # Try to read a frame
        ret, frame = camera.read()
        
        if not ret:
            print("❌ Error: Could not read frame from camera")
            camera.release()
            return False
        
        print(f"✅ Frame captured successfully - Size: {frame.shape}")
        
        # Show camera feed for a few seconds
        print("Showing camera feed for 5 seconds...")
        print("Press 'q' to quit early")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add text
            cv2.putText(frame, "Camera Test - Press 'q' to quit", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        print("✅ Camera test completed successfully!")
        print("Your camera is ready for fruit detection!")
        return True
        
    except Exception as e:
        print(f"❌ Error during camera test: {e}")
        return False

if __name__ == "__main__":
    if test_camera():
        print("\n🎉 All good! You can now run:")
        print("   python demo.py       # Quick demo")
        print("   python main.py       # Full application")
    else:
        print("\n⚠️  Camera issues detected. Please fix camera problems before proceeding.")
        sys.exit(1)
