# fast_enhanced_contextual_features.py

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import math

# === Paths ===
lane_path = "data/processed/constant_features.csv"
ego_path = "data/processed/ego_vehicle.csv"
output_path = "data/processed/contextual_features_enhanced.npy"

# === Encode lane types ===
def encode_categorical(val, vocab):
    return vocab.get(val, 0.0)

def compute_curvature(points):
    if len(points) < 3:
        return 0.0
    coords = np.array(points)
    dx = np.gradient(coords[:, 0])
    dy = np.gradient(coords[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)
    return np.mean(curvature)

# === Preparse Lanes Once ===
def load_lanes(lane_df):
    lanes_list = []
    lane_type_vocab = {}
    lane_type_counter = 1

    for _, row in lane_df.iterrows():
        try:
            left = eval(row["left_lane_boundary"])
            right = eval(row["right_lane_boundary"])
            if isinstance(left, list) and isinstance(right, list) and len(left) == len(right) and len(left) > 0:
                center = [( (lp["x"] + rp["x"]) / 2, (lp["y"] + rp["y"]) / 2 ) for lp, rp in zip(left, right)]
                center = np.array(center)

                lane_type = row.get("lane_type", "UNKNOWN")
                if lane_type not in lane_type_vocab:
                    lane_type_vocab[lane_type] = lane_type_counter / 100.0
                    lane_type_counter += 1

                lanes_list.append({
                    "center": center,
                    "type_encoded": lane_type_vocab[lane_type],
                    "curvature": compute_curvature(center)
                })
        except:
            continue

    return lanes_list

# === Main Feature Extraction ===
def extract_features():
    lane_df = pd.read_csv(lane_path)
    ego_df = pd.read_csv(ego_path)
    features = []

    print("🔵 Parsing all lanes...")
    lanes = load_lanes(lane_df)
    print(f"✅ Parsed {len(lanes)} lane segments.")

    for _, ego in tqdm(ego_df.iterrows(), total=len(ego_df)):
        ex, ey = ego["x"], ego["y"]
        vx, vy = ego.get("vx", 0.0), ego.get("vy", 0.0)
        ax, ay = ego.get("ax", 0.0), ego.get("ay", 0.0)

        closest_dist = float('inf')
        closest_alignment = 0.0
        closest_curvature = 0.0
        encoded_type = 0.0

        for lane in lanes:
            center = lane["center"]
            dists = np.sqrt((center[:, 0] - ex)**2 + (center[:, 1] - ey)**2)
            min_dist = np.min(dists)

            if min_dist < closest_dist:
                closest_dist = min_dist

                idx = np.argmin(dists)
                if idx > 0 and idx < len(center) - 1:
                    dx = center[idx + 1][0] - center[idx - 1][0]
                    dy = center[idx + 1][1] - center[idx - 1][1]
                    lane_vector = np.array([dx, dy])
                    lane_vector = lane_vector / np.linalg.norm(lane_vector) if np.linalg.norm(lane_vector) > 0 else lane_vector
                    velocity = np.array([vx, vy])
                    velocity = velocity / np.linalg.norm(velocity) if np.linalg.norm(velocity) > 0 else velocity
                    closest_alignment = np.dot(lane_vector, velocity)

                closest_curvature = lane["curvature"]
                encoded_type = lane["type_encoded"]

        weight = 1 / (closest_dist + 1e-3)  # or np.exp(-closest_dist / 20)

        features.append([
            closest_dist * weight,
            closest_alignment * weight,
            closest_curvature * weight,
            encoded_type * weight,
            vx * weight,
            vy * weight,
            ax * weight,
            ay * weight
        ])


    features = np.array(features)
    np.save(output_path, features)
    print(f"✅ Saved enhanced contextual features to {output_path}")
    print("Shape:", features.shape)

if __name__ == "__main__":
    extract_features()
