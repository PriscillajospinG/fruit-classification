# Fruit Classification with Live Camera Detection

A Python project for real-time fruit classification using computer vision and deep learning. This system can identify fruits through your camera in real-time.

## Features

- **Real-time Detection**: Live fruit classification using your webcam
- **Multiple Fruit Types**: Supports 8 different fruits (apple, banana, orange, grape, strawberry, kiwi, mango, pineapple)
- **Deep Learning**: Uses CNN (Convolutional Neural Network) for classification
- **Data Collection**: Built-in tool to collect your own training data
- **Easy to Use**: Simple command-line interface

## Project Structure

```
fruit-classification/
├── README.md
├── requirements.txt
├── main.py                 # Main script to run the system
├── src/
│   ├── fruit_detector.py   # Core detection and classification logic
│   ├── train_model.py      # Model training functionality
│   └── data_collector.py   # Data collection tool
├── models/                 # Trained models will be saved here
└── data/                   # Training data directory
    └── fruits/
        ├── apple/
        ├── banana/
        ├── orange/
        └── ...
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fruit-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start - Live Detection

Run the main script to start live fruit detection:

```bash
python main.py
```

or

```bash
python main.py live
```

This will:
- Check if a trained model exists
- If no model exists, offer to train one first
- Start the camera and begin real-time fruit detection

### Training Your Own Model

To train a model with your own data:

```bash
python main.py train
```

### Collect Your Own Data

To collect training images using your camera:

```bash
python src/data_collector.py
```

This interactive tool will help you:
- Capture images for each fruit type
- Organize them in the correct directory structure
- Build a custom dataset

## How It Works

1. **Camera Input**: Captures frames from your webcam
2. **Region of Interest**: Focuses on a specific area where you place the fruit
3. **Preprocessing**: Resizes and normalizes the image
4. **CNN Prediction**: Uses a trained neural network to classify the fruit
5. **Real-time Display**: Shows the prediction with confidence score

## Model Architecture

The CNN model includes:
- Multiple convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout layer for regularization
- Dense layers for final classification
- Data augmentation for better generalization

## Controls

During live detection:
- **Place fruit** in the green rectangle on screen
- **Press 'q'** to quit the application
- **Confidence threshold**: Only shows predictions above 50% confidence

## Supported Fruits

- Apple
- Banana
- Orange
- Grape
- Strawberry
- Kiwi
- Mango
- Pineapple

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Working webcam

## Tips for Better Results

1. **Good Lighting**: Ensure adequate lighting for clear images
2. **Clean Background**: Use a plain background when possible
3. **Multiple Angles**: Collect training data from various angles
4. **Steady Hand**: Keep the fruit steady in the detection area
5. **Training Data**: More training images = better accuracy

## Troubleshooting

### Camera Issues
- Ensure your camera is not being used by another application
- Check camera permissions in your system settings
- Try unplugging and reconnecting USB cameras

### Low Accuracy
- Collect more training data
- Ensure good lighting during data collection
- Train for more epochs
- Clean your camera lens

### Installation Issues
- Make sure you have Python 3.7 or higher
- Use a virtual environment to avoid conflicts
- On macOS, you might need to install additional packages for camera access

## Future Improvements

- [ ] Add more fruit types
- [ ] Implement object detection to find fruits in complex scenes
- [ ] Add nutritional information display
- [ ] Mobile app version
- [ ] Cloud deployment option

## Contributing

Feel free to contribute by:
- Adding new fruit types
- Improving the model architecture
- Enhancing the user interface
- Fixing bugs or adding features

## License

This project is open source and available under the MIT License.