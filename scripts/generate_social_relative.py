#!/usr/bin/env python3
"""
generate_social_vehicles_relative.py

Usage:
    python generate_social_vehicles_relative.py <ar2_root> <output_csv>

Finds every scene folder containing both
  • city_SE3_egovehicle.feather
  • annotations.feather
and for each moving vehicle writes out:
 timestamp_ns, track_uuid,
 x,y,z (ego-centric),
 vx,vy,vz, ax,ay,az,
 yaw, delta_yaw, city
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from config import PROCESSED_DATA_DIR, SENSOR_DIR

# same moving vehicle list as in the original script :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
MOVING_VEHICLES = {
    "REGULAR_VEHICLE","BUS","TRUCK","MOTORCYCLE","MOTORCYCLIST",
    "ARTICULATED_BUS","TRUCK_CAB","VEHICULAR_TRAILER"
}

def compute_velocity_and_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute vx,vy,vz and ax,ay,az from tx_m,ty_m,tz_m & timestamp_ns
    (mirrors compute_velocity_and_acceleration in the original) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}.
    """
    df = df.sort_values(['track_uuid','timestamp_ns'])
    dt = df.groupby('track_uuid')['timestamp_ns'].diff() / 1e9  # seconds

    # velocities
    df['vx'] = df.groupby('track_uuid')['tx_m'].diff() / dt
    df['vy'] = df.groupby('track_uuid')['ty_m'].diff() / dt
    df['vz'] = df.groupby('track_uuid')['tz_m'].diff() / dt

    # accelerations
    df['ax'] = df.groupby('track_uuid')['vx'].diff() / dt
    df['ay'] = df.groupby('track_uuid')['vy'].diff() / dt
    df['az'] = df.groupby('track_uuid')['vz'].diff() / dt

    # fill NaNs (first frame per track)
    for c in ('vx','vy','vz','ax','ay','az'):
        df[c] = df[c].fillna(0.0)

    return df

def compute_yaw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yaw = arctan2(dy,dx) from successive tx_m,ty_m
    (matches compute_yaw in the original) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.
    """
    df = df.sort_values(['track_uuid','timestamp_ns'])
    df['dx'] = df.groupby('track_uuid')['tx_m'].diff()
    df['dy'] = df.groupby('track_uuid')['ty_m'].diff()
    df['yaw'] = np.arctan2(df['dy'], df['dx']).fillna(0.0)
    return df

def compute_delta_yaw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute change in yaw per timestamp, normalized to [-π,π]
    (matches compute_delta_yaw in the original) :contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}.
    """
    if 'yaw' not in df.columns:
        df['delta_yaw'] = 0.0
    else:
        df['delta_yaw'] = df.groupby('track_uuid')['yaw'].diff().fillna(0.0)
        # wrap into [-π,π]
        df['delta_yaw'] = np.arctan2(
            np.sin(df['delta_yaw']),
            np.cos(df['delta_yaw'])
        )
    return df

def find_scenes(root: Path):
    """
    Walk root, yield each folder containing both required feathers
    (same as find_feather_pairs) :contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}.
    """
    for dp, dirs, files in os.walk(root):
        if 'city_SE3_egovehicle.feather' in files and 'annotations.feather' in files:
            yield Path(dp)

def collect_relative(scene_dir: Path, city: str):
    """
    From annotations.feather, filter MOVING_VEHICLES,
    read tx_m,ty_m,tz_m (already ego-centric),
    compute vx,vy,vz, ax,ay,az, yaw, delta_yaw,
    and return list of dict rows.
    """
    ann_f = scene_dir / 'annotations.feather'
    df = pd.read_feather(str(ann_f))

    # filter by label column (original script used 'label_prescribed') :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    label_col = next((c for c in df.columns if 'label' in c.lower()), None)
    if label_col:
        df = df[df[label_col].isin(MOVING_VEHICLES)].copy()

    if df.empty:
        return []

    # ensure ego-centric translations
    if not all(c in df.columns for c in ('tx_m','ty_m','tz_m')):
        raise RuntimeError(f"Missing tx_m/ty_m/tz_m in {ann_f}")

    # compute motion features
    df = compute_velocity_and_acceleration(df)
    df = compute_yaw(df)
    df = compute_delta_yaw(df)

    track_id_col = 'track_uuid' if 'track_uuid' in df.columns else 'track_id'
    rows = []
    for r in df.itertuples(index=False):
        rows.append({
            'timestamp_ns': int(r.timestamp_ns),
            'track_uuid':   getattr(r, track_id_col),
            'x':            float(r.tx_m),
            'y':            float(r.ty_m),
            'z':            float(r.tz_m),
            'vx':           float(r.vx),
            'vy':           float(r.vy),
            'vz':           float(r.vz),
            'ax':           float(r.ax),
            'ay':           float(r.ay),
            'az':           float(r.az),
            'yaw':          float(r.yaw),
            'delta_yaw':    float(r.delta_yaw),
            'city':         city
        })
    return rows

def main():

    root_dir = SENSOR_DIR
    out_csv  = PROCESSED_DATA_DIR
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for scene_dir in find_scenes(root_dir):
        # city name is the parent folder of this scene (e.g. "boston","miami","val")
        city = scene_dir.parent.name
        print(f"→ Scene {scene_dir.name} (city={city})")

        recs = collect_relative(scene_dir, city)
        print(f"   • {len(recs)} actor-frames")
        all_rows.extend(recs)

    if not all_rows:
        print("❌ No records extracted—check your paths.")
        sys.exit(1)

    df_out = pd.DataFrame(all_rows)
    print(f"Writing {len(df_out)} rows to {out_csv}")
    df_out.to_csv(os.path.join(out_csv, 'social_vehicles_relative.csv'), index=False)
    print("✅ Done.")

if __name__ == '__main__':
    main()
