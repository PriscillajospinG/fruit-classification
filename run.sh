#!/bin/bash

# Fruit Classification Project Helper Script

PYTHON_CMD="/Users/priscillajosping/Downloads/fruit-classification/.venv/bin/python"

show_help() {
    echo "🍎 Fruit Classification Helper 🍌"
    echo "================================="
    echo ""
    echo "Usage: ./run.sh <command>"
    echo ""
    echo "Commands:"
    echo "  test-camera    Test if camera is working"
    echo "  demo          Run quick demo with synthetic data"
    echo "  live          Start live fruit detection"
    echo "  train         Train model with real data"
    echo "  collect       Collect training images"
    echo "  install       Install Python dependencies"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh test-camera    # Test camera first"
    echo "  ./run.sh demo          # Quick demo"
    echo "  ./run.sh live          # Start detection"
}

case "$1" in
    "test-camera")
        echo "Testing camera..."
        $PYTHON_CMD test_camera.py
        ;;
    "demo")
        echo "Running demo..."
        $PYTHON_CMD demo.py
        ;;
    "live")
        echo "Starting live detection..."
        $PYTHON_CMD main.py live
        ;;
    "train")
        echo "Training model..."
        $PYTHON_CMD main.py train
        ;;
    "collect")
        echo "Starting data collection..."
        $PYTHON_CMD src/data_collector.py
        ;;
    "install")
        echo "Installing dependencies..."
        $PYTHON_CMD -m pip install -r requirements.txt
        ;;
    "help"|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
