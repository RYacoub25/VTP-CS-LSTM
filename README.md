Vehicle Trajectory Prediction with Contextual Social LSTMs (CS-LSTM)
Overview
This project implements a Contextual Social LSTM model to predict future trajectories of target vehicles based on ego vehicle data, neighboring vehicles, and contextual lane information using the Argoverse 2 dataset. The model predicts incremental displacements over a sequence to anticipate future vehicle positions and improve autonomous driving decisions.

Features
Predict trajectories of vehicles near the ego vehicle within a configurable radius.

Integrates social interactions via neighboring vehicle history.

Incorporates contextual lane and map features.

Supports variable prediction length (e.g., 30 timesteps / 3 seconds).

Visualization tools for trajectories, lane boundaries, map objects, and vehicle positions.

Getting Started
Prerequisites
Python 3.9+

Conda for environment management

CUDA (optional, if using GPU acceleration)

Setup Environment
Use the provided environment.yml to set up your conda environment:

bash
Copy
Edit
conda env create -f environment.yml
conda activate av2-py39
Dataset Preparation
Ensure you have the Argoverse 2 dataset downloaded and processed. Place the processed CSV and numpy files in the data/processed/ folder:

ego_vehicle_with_intention.csv

social_vehicles_relative.csv

contextual_features_merged.npy

constant_features.csv (lane boundaries)

map_objects.csv (map objects)

Training
Train the model with default parameters or customize in train.py:

bash
Copy
Edit
python CS_LSTM/train.py --use_delta_yaw --use_intention
You can also specify parameters like:

bash
Copy
Edit
python CS_LSTM/train.py --seq_len 30 --pred_len 30 --batch 64 --epochs 20 --use_delta_yaw
Visualization
Visualize predictions and map context using:

bash
Copy
Edit
python visualize/visualize_everything.py
This script loads the trained model and displays interactive plots with:

Ego vehicle (car icon)

Target vehicle (purple dot)

History, ground truth, and predicted trajectories

Lane boundaries and map objects

Project Structure
graphql
Copy
Edit
├── CS_LSTM/
│   ├── dataset.py         # Dataset loader and pre-processing
│   ├── model.py           # Contextual Social LSTM model definition
│   ├── train.py           # Training script
├── visualize/
│   ├── visualize_everything.py  # Visualization of predictions & map context
├── data/
│   ├── processed/         # Preprocessed CSVs and numpy feature files
├── environment.yml        # Conda environment configuration
├── README.md              # This file
Important Notes
Ensure consistent use of use_delta_yaw flag during training and visualization to match input feature dimensions.

Relative vehicle positions are computed with respect to the ego vehicle for accurate social context.

Incremental displacements are predicted and then converted to global positions for visualization.

Visualization uses matplotlib and supports lane boundaries, map objects, and vehicle icons.

Citation
If you use this code or model for research, please cite the relevant papers on Social LSTMs and contextual trajectory prediction.

License
Specify your license here (e.g., MIT, Apache 2.0).