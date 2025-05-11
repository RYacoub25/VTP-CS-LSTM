# config.py
from pathlib import Path

# Path to Argoverse 2 root directory (adjust to your system)
ARGO_DATA_PATH = "G:/My Drive/Argoverse 2 Dataset"  # Example; update with your actual path

# Example subfolder for motion forecasting (you can customize later)
MOTION_FORECASTING_DIR = Path(f"{ARGO_DATA_PATH}/Motion Forecasting Dataset/Validation")
SENSOR_DIR = Path(f"{ARGO_DATA_PATH}/Sensor Dataset/Validation Part 1/sensor/val")
PROCESSED_DATA_DIR = Path("data/processed")

# Time settings
SEQUENCE_LENGTH = 20  # 2s at 10Hz
PREDICTION_HORIZON = 30  # e.g., 3s prediction