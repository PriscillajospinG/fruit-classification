#!/usr/bin/env python3
"""
Main script to run the fruit classification system
"""

import sys
import os
import argparse
from src.fruit_detector import FruitDetector
from src.train_model import FruitTrainer

def train_model():
    """Train a new fruit classification model"""
    print("Training options:")
    print("1. Train with Fruits-360 dataset (recommended)")
    print("2. Train with synthetic data (demo)")
    
    choice = input("Choose training option (1/2): ").strip()
    
    if choice == "1":
        # Train with Fruits-360 dataset
        print("Training with Fruits-360 dataset...")
        try:
            from train_fruits360 import main as train_fruits360
            train_fruits360()
            return
        except Exception as e:
            print(f"Error training with Fruits-360: {e}")
            print("Falling back to synthetic data training...")
    
    # Fallback to synthetic data training
    print("Training with synthetic data...")
    
    trainer = FruitTrainer("data/fruits")
    
    try:
        # Try to load real data first, fallback to synthetic
        trainer.prepare_data(use_synthetic=False)
    except:
        print("No real data found, using synthetic data for training...")
        trainer.prepare_data(use_synthetic=True)
    
    # Train the model
    history = trainer.train_model(epochs=20)
    
    # Evaluate
    trainer.evaluate_model()
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    trainer.classifier.save_model("models/fruit_classifier.h5")
    
    print("Model training completed and saved to models/fruit_classifier.h5")

def run_live_detection():
    """Run live fruit detection from camera"""
    print("Starting live fruit detection...")
    
    # Check for Fruits-360 model first
    fruits360_model = "models/fruits360_classifier.h5"
    basic_model = "models/fruit_classifier.h5"
    demo_model = "models/fruit_classifier_demo.h5"
    
    model_path = None
    
    if os.path.exists(fruits360_model):
        print("Found Fruits-360 trained model - using advanced detection")
        try:
            from fruits360_detector import Fruits360Detector
            detector = Fruits360Detector(fruits360_model)
            detector.detect_fruits_live()
            return
        except Exception as e:
            print(f"Error with Fruits-360 detector: {e}")
            print("Falling back to basic detector...")
    
    # Use basic detector
    if os.path.exists(basic_model):
        model_path = basic_model
    elif os.path.exists(demo_model):
        model_path = demo_model
    else:
        print("No trained model found!")
        response = input("Would you like to train a model first? (y/n): ")
        if response.lower() == 'y':
            train_model()
            # Try again after training
            if os.path.exists(fruits360_model):
                model_path = fruits360_model
            elif os.path.exists(basic_model):
                model_path = basic_model
        else:
            print("Running with untrained model (predictions will be random)")
    
    try:
        detector = FruitDetector(model_path)
        detector.detect_fruits_live()
    except Exception as e:
        print(f"Error running detection: {e}")
        print("Make sure your camera is connected and accessible.")

def main():
    parser = argparse.ArgumentParser(description="Fruit Classification System")
    parser.add_argument(
        'command', 
        choices=['train', 'detect', 'live', 'train-fruits360'],
        help='Command to run: train (train model), detect/live (live detection), train-fruits360 (train with Fruits-360 dataset)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command == 'train-fruits360':
        from train_fruits360 import main as train_fruits360
        train_fruits360()
    elif args.command in ['detect', 'live']:
        run_live_detection()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, default to live detection
        print("Fruit Classification System")
        print("=" * 30)
        print("Available commands:")
        print("  python main.py train-fruits360  # Train with Fruits-360 dataset")
        print("  python main.py train           # Train with other data")
        print("  python main.py live            # Live detection")
        print()
        
        choice = input("What would you like to do? (train-fruits360/train/live): ").strip().lower()
        
        if choice in ['train-fruits360', 'fruits360', '1']:
            from train_fruits360 import main as train_fruits360
            train_fruits360()
        elif choice in ['train', '2']:
            train_model()
        else:
            run_live_detection()
    else:
        main()
