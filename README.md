# EyeMe: Intelligent Camera for Your Eyes

EyeMe is a smart glasses project that monitors eye health metrics such as blink rate, redness, and pupil dilation using a Raspberry Pi Zero 2 W and a Pi Camera Module V2.

## Features

- Real-time blink detection
- Redness analysis
- Pupil dilation measurement
- Face recognition with snapshot labeling
- Streamlit dashboard for data visualization

## Hardware Requirements

- Raspberry Pi Zero 2 W
- Raspberry Pi Camera Module V2
- MicroSD Card (32GB recommended)
- Power supply for Raspberry Pi

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/realRSB/EyeMe.git
   cd ICY

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Run the main application:
    ```bash
    python main.py

2. launch the dashboard:
    ```bash
    streamlit run dashboard.py

