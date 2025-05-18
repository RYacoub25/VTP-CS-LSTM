# Re-importing necessary libraries after kernel reset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === File paths ===
ego_path = "data/processed/ego_vehicle_with_intention.csv"
social_path = "data/processed/social_vehicle_fixed3.csv"
lane_path = "data/processed/constant_features.csv"
map_obj_path = "data/processed/map_objects.csv"

# === Load data ===
ego_df = pd.read_csv(ego_path)
social_df = pd.read_csv(social_path)
lane_df = pd.read_csv(lane_path)
map_df = pd.read_csv(map_obj_path)

# === Map object types of interest ===
TARGET_CLASSES = {
    "STOP_SIGN": "red",
    "SIGN": "orange",
    "TRAFFIC_LIGHT_TRAILER": "purple",
    "CONSTRUCTION_CONE": "blue",
    "CONSTRUCTION_BARREL": "brown",
    "MESSAGE_BOARD_TRAILER": "pink",
    "MOBILE_PEDESTRIAN_SIGN": "cyan"
}

# === Find a timestamp with social vehicles near ego ===
def find_valid_timestamp(radius=50):
    common_ts = np.intersect1d(ego_df["timestamp_ns"].unique(), social_df["timestamp_ns"].unique())
    for ts in common_ts:
        ego = ego_df[ego_df["timestamp_ns"] == ts]
        social = social_df[social_df["timestamp_ns"] == ts]
        if ego.empty or social.empty:
            continue
        ex, ey = ego.iloc[0]["x"], ego.iloc[0]["y"]
        sx, sy = social["x"].values, social["y"].values
        dists = np.sqrt((sx - ex) ** 2 + (sy - ey) ** 2)
        if (dists < radius).any():
            return ts
    return None

# === Plot scene at timestamp ===
def plot_scene(ts):
    ego = ego_df[ego_df["timestamp_ns"] == ts].iloc[0]
    ex, ey = ego["x"], ego["y"]

    plt.figure(figsize=(12, 12))
    plt.scatter(ex, ey, c='red', s=80, label="Ego Vehicle" )

    social = social_df[social_df["timestamp_ns"] == ts]
    plt.scatter(social["x"], social["y"], c='blue', label="Social Vehicles", alpha=0.7)

    for _, row in lane_df.iterrows():
        try:
            left = eval(row["left_lane_boundary"])
            right = eval(row["right_lane_boundary"])
            if isinstance(left, list) and len(left) > 1:
                lx = [p["x"] for p in left]
                ly = [p["y"] for p in left]
                plt.plot(lx, ly, c='green', linewidth=1, alpha=0.4)
            if isinstance(right, list) and len(right) > 1:
                rx = [p["x"] for p in right]
                ry = [p["y"] for p in right]
                plt.plot(rx, ry, c='black', linewidth=1, alpha=0.4)
        except:
            continue

    map_ts = map_df[map_df["timestamp_ns"] == ts]
    for cls, color in TARGET_CLASSES.items():
        cls_objs = map_ts[map_ts["category"] == cls]
        if not cls_objs.empty:
            plt.scatter(cls_objs["x"], cls_objs["y"], c=color, label=cls.replace("_", " ").title(), alpha=0.8)

    
    plt.title(f"Scene at Timestamp: {ts}")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.xlim(ex - 30, ex + 30)
    plt.ylim(ey - 30, ey + 30)
    plt.show()

# === Run ===
if __name__ == "__main__":
    ts = find_valid_timestamp()
    if ts is not None:
        plot_scene(ts)
    else:
        print("No valid timestamp found.")