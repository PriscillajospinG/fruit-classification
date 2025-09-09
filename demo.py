#!/usr/bin/env python3
"""
Demo script to quickly test the fruit classification system
"""

import os
import sys

def demo():
    """Run a quick demo of the system"""
    print("🍎 Fruit Classification Demo 🍌")
    print("=" * 40)
    
    print("\n1. First, let's train a model with synthetic data...")
    print("   (This will create a basic model for demonstration)")
    
    # Import after setting up the path
    from src.train_model import FruitTrainer
    
    trainer = FruitTrainer("data/fruits")
    
    # Create synthetic training data
    trainer.prepare_data(use_synthetic=True)
    
    # Quick training with fewer epochs for demo
    print("\n2. Training model (this may take a few minutes)...")
    history = trainer.train_model(epochs=5, batch_size=16)
    
    # Evaluate
    print("\n3. Evaluating model...")
    trainer.evaluate_model()
    
    # Save model
    os.makedirs("models", exist_ok=True)
    trainer.classifier.save_model("models/fruit_classifier_demo.h5")
    
    print("\n4. Model saved! Now you can run live detection:")
    print("   python main.py live")
    print("\n   Or train with real data:")
    print("   python src/data_collector.py  # Collect your own images")
    print("   python main.py train          # Train with real data")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("- Run 'python main.py live' to start camera detection")
    print("- Use 'python src/data_collector.py' to collect real fruit images")
    print("- Train with real data for better accuracy")

if __name__ == "__main__":
    try:
        demo()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all packages are installed: pip install -r requirements.txt")
        print("2. Check if your camera is working")
        print("3. Ensure you have enough disk space")
