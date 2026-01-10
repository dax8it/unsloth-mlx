#!/bin/bash

# Unsloth-MLX GUI Launcher
# Run this script to start the Gradio GUI

echo "========================================"
echo "Unsloth-MLX GUI"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if unsloth_mlx is installed
if ! python3 -c "import unsloth_mlx" 2>/dev/null; then
    echo ""
    echo "Error: Unsloth-MLX not installed!"
    echo "Please install it first:"
    echo "  pip install -e ."
    exit 1
fi

# Check if gradio is installed
if ! python3 -c "import gradio" 2>/dev/null; then
    echo ""
    echo "Error: Gradio not installed!"
    echo "Please install it first:"
    echo "  pip install gradio"
    exit 1
fi

echo ""
echo "Starting Gradio GUI..."
echo ""
echo "Open your browser to: http://127.0.0.1:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Launch the GUI
python3 gui.py