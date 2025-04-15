# app/utils/logging_helpers.py

import os
import csv
from datetime import datetime

def ensure_logs_folder():
    """Create logs/ folder if it doesn't exist."""
    os.makedirs("data/logs", exist_ok=True)

def create_timestamped_log_file():
    """Return a new timestamped log filename inside /logs/."""
    timestamp = datetime.now().strftime("eye_health_log_%Y-%m-%d_%H-%M-%S.csv")
    return os.path.join("data/logs", timestamp)

def log_data(filename, data):
    """
    Appends a row of data (a dictionary) to a CSV file.
    If the file doesn't exist, writes a header first.
    """
    write_header = not os.path.exists(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)
