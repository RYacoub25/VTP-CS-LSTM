import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Inputs
ego_path = "data/processed/ego_vehicle.csv"
objects_path = "data/processed/map_objects.csv"
output_path = "data/processed/contextual_objects.npy"

# Classes to count around the ego vehicle
TARGET_CLASSES = {
    "STOP_SIGN": 0,
    "SIGN": 1,
    "TRAFFIC_LIGHT_TRAILER": 2,
    "CONSTRUCTION_CONE": 3,
    "CONSTRUCTION_BARREL": 4,
    "MESSAGE_BOARD_TRAILER": 5,
    "MOBILE_PEDESTRIAN_SIGN": 6
}

def extract_contextual_objects(ego_df, obj_df, radius=25):
    # Filter only target classes once
    obj_df = obj_df[obj_df["category"].isin(TARGET_CLASSES.keys())].copy()
    
    # Replace category with corresponding class index for faster processing
    obj_df["class_idx"] = obj_df["category"].map(TARGET_CLASSES)

    # Convert to NumPy arrays for speed
    ego_xy = ego_df[["x", "y"]].values
    obj_xy = obj_df[["x", "y"]].values
    obj_class = obj_df["class_idx"].values

    result = np.zeros((len(ego_xy), len(TARGET_CLASSES)))

    for i, (ex, ey) in enumerate(tqdm(ego_xy, desc="Computing distances")):
        dists = np.sqrt((obj_xy[:, 0] - ex)**2 + (obj_xy[:, 1] - ey)**2)
        nearby_classes = obj_class[dists < radius]

        # Count occurrences per class
        for idx, cls in enumerate(obj_class):
            if dists[idx] < radius:
                weight = 1 / (dists[idx] + 1e-3)  # or np.exp(-dists[idx] / 20)
                result[i, cls] += weight


    return result

def main():
    if not os.path.exists(ego_path) or not os.path.exists(objects_path):
        print("❌ Required CSVs not found.")
        return

    ego_df = pd.read_csv(ego_path)
    obj_df = pd.read_csv(objects_path)

    features = extract_contextual_objects(ego_df, obj_df)
    print("✅ Feature shape:", features.shape)

    np.save(output_path, features)
    print(f"✅ Contextual object features saved: {output_path}")

if __name__ == "__main__":
    main()
