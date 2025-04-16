#!/bin/bash

echo "🔧 Updating system and installing required system packages..."
sudo apt update && sudo apt install -y \
  python3 python3-pip python3-venv git \
  libatlas-base-dev libjpeg-dev libtiff5 \
  libopenblas-dev libhdf5-dev libavformat-dev \
  libqtgui4 python3-pyqt5 libilmbase-dev \
  libgtk-3-dev libgl1-mesa-glx

echo "📦 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "⬇️ Installing Eyeme dependencies (lightweight Pi version)..."
pip install --upgrade pip
pip install -r requirements-pi.txt

echo "🧠 Installing Eyeme as editable Python package..."
pip install -e .

echo "✅ Setup complete! You can now run ./scripts/start.sh"
