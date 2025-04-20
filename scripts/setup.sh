#!/bin/bash

echo "🤖 Detecting Raspberry Pi model..."
PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
echo "📟 Detected: $PI_MODEL"

echo "🔧 Updating system and installing required system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system packages
sudo apt-get install -y python3-opencv opencv-data mpg321 python3-pip

# Install mediapipe for Pi Zero 2 W 
echo "📦 Installing mediapipe-rpi3..."
pip3 install mediapipe-rpi3

# Other Python deps
echo "🔠 Installing text-to-speech..."
pip3 install gtts

echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "🧠 Installing Eyeme as editable Python package..."
pip install -e .

echo "✅ Setup complete! You can now run ./scripts/start.sh"
