#!/bin/bash

echo "🤖 Detecting Raspberry Pi model..."
PI_MODEL=$(tr -d '\0' < /proc/device-tree/model)
echo "📟 Detected: $PI_MODEL"

echo "🔧 Fixing package manager issues if any..."
sudo dpkg --configure -a

echo "🔧 Updating system and installing required system packages..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-opencv opencv-data mpg321 python3-pip python3-venv

echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📦 Installing Python packages inside virtual environment..."
pip install mediapipe-rpi3
pip install gtts

echo "🧠 Installing Eyeme as editable Python package..."
pip install -e .

echo "✅ Setup complete! You can now run ./scripts/start.sh"
