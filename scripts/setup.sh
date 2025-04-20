#!/bin/bash

echo "🔧 Updating system and installing required system packages..."
sudo apt-get update && sudo apt-get upgrade \
  apt-get install python-opencv python3-opencv opencv-data \
  pip3 install mediapipe-rpi3 \
  pip3 install mediapipe-rpi4 \
  pip3 install gtts \
  apt install mpg321 \

echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "🧠 Installing Eyeme as editable Python package..."
pip install -e .

echo "✅ Setup complete! You can now run ./scripts/start.sh"
