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
    print("Training new fruit classification model...")
    
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
    
    model_path = "models/fruit_classifier.h5"
    
    if not os.path.exists(model_path):
        print(f"No trained model found at {model_path}")
        response = input("Would you like to train a model first? (y/n): ")
        if response.lower() == 'y':
            train_model()
        else:
            print("Running with untrained model (predictions will be random)")
            model_path = None
    
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
        choices=['train', 'detect', 'live'],
        help='Command to run: train (train model), detect/live (live detection)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model()
    elif args.command in ['detect', 'live']:
        run_live_detection()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, default to live detection
        print("No command specified. Starting live detection...")
        print("Usage: python main.py [train|detect|live]")
        print("- train: Train a new model")
        print("- detect/live: Start live fruit detection")
        print()
        run_live_detection()
    else:
        main()
