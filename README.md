# Fruit Classification with Live Camera Detection

Real-time fruit classification using computer vision and deep learning. Identifies fruits through your webcam using the professional Fruits-360 dataset with 98%+ accuracy.

## 🍎 Supported Fruits
96+ fruit varieties including:
- **Apples**: Golden, Red, Granny Smith, Braeburn, Pink Lady, Crimson Snow
- **Bananas**: Regular, Lady Finger, Red Banana
- **Citrus**: Orange, Lemon, Grapefruit (Pink/White)
- **Grapes**: Blue, Pink, White varieties
- **Berries**: Strawberry, Cherry varieties (Rainier, Wax, Sour)
- **Tropical**: Mango, Pineapple, Kiwi, Avocado
- **Stone Fruits**: Peach, Pear varieties, Plum
- And many more!

## 🚀 Quick Start (3 Steps)

### Step 1: Setup Environment
```bash
# Navigate to project directory
cd fruit-classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Train Model with Real Dataset
```bash
python train_fruits360.py
```
**Training Results**: ~98% accuracy, 96 fruit classes, 15-30 minutes

### Step 3: Start Live Detection
```bash
python main.py live
```
**Controls**: Place fruit in green box, press 'q' to quit, 's' to save, 'p' for top predictions
Apple, Banana, Orange, Grape, Strawberry, Kiwi, Mango, Pineapple

## Quick Start

1. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   
   # On Windows:
   # .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

4. **When done, deactivate virtual environment**:
   ```bash
   deactivate
   ```

## Usage Options

**Note**: Make sure your virtual environment is activated before running any commands:
```bash
source .venv/bin/activate  # On macOS/Linux
```

### 1. Train with Fruits-360 Dataset (Recommended)
```bash
python main.py train-fruits360
# or
python train_fruits360.py
```
Uses the real Fruits-360 dataset for high-accuracy training.

### 2. Live Detection (Default)
```bash
python main.py
# or
python main.py live
```
- Points camera at fruit
- Places fruit in green rectangle on screen
- Press 'q' to quit

### 2. Train Custom Model
```bash
python main.py train
```
Trains model with synthetic data (for demo) or your collected images.

### 3. Collect Training Data
```bash
python src/data_collector.py
```
Interactive tool to capture fruit images with your camera.

### 4. Quick Demo
```bash
python demo.py
```
Trains a basic model and shows how to use the system.

## Requirements

- Python 3.7+
- Working webcam
- Virtual environment (recommended)
- Dependencies in `requirements.txt`

## Installation Steps

1. **Clone or download the project**
2. **Navigate to project directory**:
   ```bash
   cd fruit-classification
   ```
3. **Create virtual environment**:
   ```bash
   python -m venv .venv
   ```
4. **Activate virtual environment**:
   ```bash
   # macOS/Linux:
   source .venv/bin/activate
   
   # Windows:
   .venv\Scripts\activate
   ```
5. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## How It Works

1. Captures video from webcam
2. Processes images through CNN model
3. Displays fruit name and confidence score
4. Shows predictions above 50% confidence

## Project Structure

```
fruit-classification/
├── main.py              # Main application
├── demo.py              # Quick demo
├── requirements.txt     # Dependencies
├── src/
│   ├── fruit_detector.py    # Detection logic
│   ├── train_model.py       # Model training
│   └── data_collector.py    # Data collection
├── models/              # Saved models
└── data/               # Training images
```

## Troubleshooting

**Camera not working:**
- Close other apps using camera
- Check camera permissions
- Try `python test_camera.py`

**Low accuracy:**
- Use good lighting
- Collect more training images
- Ensure clean background

**Installation issues:**
- Use Python virtual environment (recommended)
- Update pip: `pip install --upgrade pip`
- Make sure virtual environment is activated
- On macOS, may need: `brew install python-tk`

**Virtual environment issues:**
- If activation fails, try: `python3 -m venv .venv`
- Ensure you're in the project directory
- Use `which python` to verify correct Python version