#!/bin/bash

echo "🔧 Updating system and installing required system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install required system packages
sudo apt-get install -y python3-opencv opencv-data mpg321

# Install Python packages
pip3 install mediapipe-rpi3
pip3 install mediapipe-rpi4
pip3 install gtts

echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "🧠 Installing Eyeme as editable Python package..."
pip install -e .

echo "✅ Setup complete! You can now run ./scripts/start.sh"
