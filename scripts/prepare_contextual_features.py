import pandas as pd
import numpy as np
import ast
import os
from sklearn.neighbors import NearestNeighbors

input_path = "data/processed/constant_features.csv"
ego_path = "data/processed/ego_vehicle.csv"
output_path = "data/processed/contextual_features.npy"

def parse_coords(coord_list):
    if isinstance(coord_list, str):
        try:
            parsed = ast.literal_eval(coord_list)
            if isinstance(parsed, list) and len(parsed) > 0:
                coords = np.array([[pt['x'], pt['y'], pt['z']] for pt in parsed if isinstance(pt, dict) and 'x' in pt])
                return coords
        except Exception as e:
            print(f"❌ Failed to parse: {coord_list} → {e}")
    return None

def extract_features():
    if not os.path.exists(input_path) or not os.path.exists(ego_path):
        print("❌ Required file(s) not found.")
        return

    lanes_df = pd.read_csv(input_path)
    ego_df = pd.read_csv(ego_path)

    lane_centers = []
    lane_meta = []

    for _, row in lanes_df.iterrows():
        left = parse_coords(row.get("left_lane_boundary"))
        right = parse_coords(row.get("right_lane_boundary"))

        if left is None or right is None or len(left) != len(right):
            continue  # Skip inconsistent lane boundaries

        center = ((left[:len(right)] + right) / 2).mean(axis=0)
        lane_centers.append(center[:2])  # only x, y

        meta = [
            hash(row.get("lane_type", "UNKNOWN")) % 1000 / 1000.0,
            hash(row.get("left_lane_mark_type", "UNKNOWN")) % 1000 / 1000.0,
            hash(row.get("right_lane_mark_type", "UNKNOWN")) % 1000 / 1000.0
        ]
        lane_meta.append(meta)

    if len(lane_centers) == 0:
        print("❌ No valid lane segments found.")
        return

    nbrs = NearestNeighbors(n_neighbors=1).fit(np.array(lane_centers))
    ego_xy = ego_df[['x', 'y']].values
    distances, indices = nbrs.kneighbors(ego_xy)

    context_features = []
    for idx in indices.flatten():
        context_features.append(np.concatenate([lane_centers[idx], lane_meta[idx]]))

    final = np.array(context_features)
    np.save(output_path, final)
    print(f"✅ Contextual features saved to: {output_path}")

if __name__ == "__main__":
    extract_features()

    print("✅ All done.")