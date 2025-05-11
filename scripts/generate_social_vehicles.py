#!/usr/bin/env python3
"""
generate_social_vehicles_from_feather.py

Usage:
    python generate_social_vehicles_from_feather.py /path/to/motion_forecasting_root output_social.csv
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from config import PROCESSED_DATA_DIR, SENSOR_DIR

# Define the relevant vehicle categories
MOVING_VEHICLES = [
    "REGULAR_VEHICLE", "BUS", "TRUCK", "MOTORCYCLE", "MOTORCYCLIST", 
    "ARTICULATED_BUS", "TRUCK_CAB", "VEHICULAR_TRAILER"
]

def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert unit quaternion into a 3×3 rotation matrix."""
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def find_feather_pairs(root_dir):
    """Yield (se3_feather, annotations_feather) for each scenario folder."""
    for dp, _, files in os.walk(root_dir):
        if 'city_SE3_egovehicle.feather' in files and 'annotations.feather' in files:
            yield Path(dp)/'city_SE3_egovehicle.feather', Path(dp)/'annotations.feather'

def load_se3_lookup(se3_path):
    """
    Reads city_SE3_egovehicle.feather, finds:
      - timestamp_ns (or timestamp)
      - quaternion qw,qx,qy,qz
      - translation tx_m,ty_m,tz_m
    Returns dict: ts -> (R, t) where R is 3×3, t is (3,).
    """
    df = pd.read_feather(se3_path)
    ts_col = 'timestamp_ns' if 'timestamp_ns' in df.columns else 'timestamp'

    qc = {}; tc = {}
    for c in df.columns:
        lc = c.lower()
        if   lc in ('qw','rot_w','quaternion_w'):    qc['w'] = c
        elif lc in ('qx','rot_x','quaternion_x'):    qc['x'] = c
        elif lc in ('qy','rot_y','quaternion_y'):    qc['y'] = c
        elif lc in ('qz','rot_z','quaternion_z'):    qc['z'] = c
        elif lc in ('tx','tx_m','translation_x'):     tc['x'] = c
        elif lc in ('ty','ty_m','translation_y'):     tc['y'] = c
        elif lc in ('tz','tz_m','translation_z'):     tc['z'] = c

    if set(qc.keys()) != {'w','x','y','z'} or set(tc.keys()) != {'x','y','z'}:
        raise RuntimeError(f"Cannot detect SE3 columns in {se3_path}\n  qc={qc}\n  tc={tc}")

    lookup = {}
    for row in df.itertuples(index=False):
        ts = getattr(row, ts_col)
        w = getattr(row, qc['w']); x = getattr(row, qc['x'])
        y = getattr(row, qc['y']); z = getattr(row, qc['z'])
        tx = getattr(row, tc['x']); ty = getattr(row, tc['y']); tz = getattr(row, tc['z'])
        R = quaternion_to_rotation_matrix(w, x, y, z)
        t = np.array([tx, ty, tz], dtype=float)
        lookup[ts] = (R, t)
    return lookup

def compute_yaw(df):
    """
    Compute yaw (orientation) for each vehicle based on changes in position.
    The yaw is the angle of the vehicle's movement direction.
    """
    # Sort by track_uuid and timestamp to ensure correct ordering per vehicle
    df = df.sort_values(by=['track_uuid', 'timestamp_ns'])

    # Check if the correct position columns exist (tx_m, ty_m, tz_m)
    if all(col in df.columns for col in ['tx_m', 'ty_m', 'tz_m']):
        df['dx'] = df.groupby('track_uuid')['tx_m'].diff()
        df['dy'] = df.groupby('track_uuid')['ty_m'].diff()
    else:
        raise RuntimeError(f"Position columns ('tx_m', 'ty_m', 'tz_m') not found in {df.columns.tolist()}")

    # Compute yaw from the arctangent of the change in position (dy/dx)
    df['yaw'] = np.arctan2(df['dy'], df['dx'])

    # Fill NaN values (for the first row and possible division by zero)
    df['yaw'] = df['yaw'].fillna(0)  # Assuming 0 yaw for the first frame or missing data
    
    return df


def compute_delta_yaw(df):
    """
    Compute the change in yaw (delta_yaw) for each vehicle.
    Assumes df is sorted by track_uuid and timestamp_ns.
    """
    if 'yaw' in df.columns:
        # Compute delta yaw only if 'yaw' column exists
        df['delta_yaw'] = df.groupby('track_uuid')['yaw'].diff().fillna(0)
    else:
        # If 'yaw' column is missing, set delta_yaw to NaN
        df['delta_yaw'] = np.nan

    # Normalize delta yaw to the range of -pi to pi for consistent direction of turn
    df['delta_yaw'] = np.arctan2(np.sin(df['delta_yaw']), np.cos(df['delta_yaw']))
    
    return df


def compute_velocity_and_acceleration(df):
    """
    Compute the velocity and acceleration for each vehicle based on position data.
    Velocity is calculated as the change in position (dx, dy, dz) over time.
    Acceleration is calculated as the change in velocity over time.
    """
    # Sort by track_uuid and timestamp to ensure correct ordering per vehicle
    df = df.sort_values(by=['track_uuid', 'timestamp_ns'])
    
    # Compute the difference in position between consecutive timestamps
    df['dx'] = df.groupby('track_uuid')['tx_m'].diff()
    df['dy'] = df.groupby('track_uuid')['ty_m'].diff()
    df['dz'] = df.groupby('track_uuid')['tz_m'].diff()

    # Compute velocity (change in position / change in time)
    df['vx'] = df.groupby('track_uuid')['dx'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
    df['vy'] = df.groupby('track_uuid')['dy'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
    df['vz'] = df.groupby('track_uuid')['dz'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()

    # Compute acceleration (change in velocity / change in time)
    df['ax'] = df.groupby('track_uuid')['vx'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
    df['ay'] = df.groupby('track_uuid')['vy'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()
    df['az'] = df.groupby('track_uuid')['vz'].diff() / df.groupby('track_uuid')['timestamp_ns'].diff()

    # Fill NaN values (for the first row and possible division by zero)
    df['vx'] = df['vx'].fillna(0)
    df['vy'] = df['vy'].fillna(0)
    df['vz'] = df['vz'].fillna(0)
    df['ax'] = df['ax'].fillna(0)
    df['ay'] = df['ay'].fillna(0)
    df['az'] = df['az'].fillna(0)

    return df



def collect_records(se3_lookup, ann_path, city_name):
    """
    Reads annotations.feather, auto-detects which columns hold
    3-vector translation, velocity, and acceleration (nested or separate),
    projects translation into global coords, carries over vx,vy,vz, ax,ay,az, delta_yaw,
    and returns a list of dict rows.
    """
    df = pd.read_feather(ann_path)

    # --- detect translation vector column ---
    if 'translation' in df.columns and hasattr(df.at[0, 'translation'], '__len__'):
        trans_arr = np.vstack(df['translation'].map(lambda x: np.array(x, float)))
    elif all(c in df.columns for c in ('tx_m', 'ty_m', 'tz_m')):
        trans_arr = df[['tx_m', 'ty_m', 'tz_m']].to_numpy(dtype=float)
    elif all(c in df.columns for c in ('x_raw', 'y_raw', 'z_raw')):
        trans_arr = df[['x_raw', 'y_raw', 'z_raw']].to_numpy(dtype=float)
    else:
        raise RuntimeError(f"No translation vector found in {ann_path}\nColumns: {df.columns.tolist()}")

    # --- Compute velocity and acceleration ---
    df = compute_velocity_and_acceleration(df)

    # --- yaw computation --- 
    df = compute_yaw(df)

    # --- delta yaw computation ---
    df = compute_delta_yaw(df)

    # track identifier
    track_col = 'track_uuid' if 'track_uuid' in df.columns else 'track_id'
    track_ids = df[track_col].to_numpy()

    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        ts = row.timestamp_ns
        st = se3_lookup.get(ts)
        if st is None:
            continue
        R, t = st
        glob = R.dot(trans_arr[i]) + t
        rows.append({
            'timestamp_ns': ts,
            'track_uuid':   track_ids[i],
            'x':            float(glob[0]),
            'y':            float(glob[1]),
            'z':            float(glob[2]),
            'vx':           float(row.vx),
            'vy':           float(row.vy),
            'vz':           float(row.vz),
            'ax':           float(row.ax),
            'ay':           float(row.ay),
            'az':           float(row.az),
            'delta_yaw':    float(row.delta_yaw),
            'yaw':          float(row.yaw),
            'city':         city_name
        })
    return rows



def main():

    root_dir, out_csv = SENSOR_DIR, PROCESSED_DATA_DIR

    all_rows = []
    for se3_f, ann_f in find_feather_pairs(root_dir):
        city = se3_f.parent.parent.name
        print(f"→ Processing {se3_f.parent} (city={city})")
        se3_lkp = load_se3_lookup(se3_f)
        recs    = collect_records(se3_lkp, ann_f, city)
        print(f"   → {len(recs)} actor-frames")
        all_rows.extend(recs)

    if not all_rows:
        print("❌ No records found under", root_dir)
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    print(f"Collected {len(df)} rows; writing {out_csv} …")
    df.to_csv(os.path.join(out_csv, 'social_vehicle_fixed3.csv'), index=False)
    print("✅ Done.")

if __name__ == '__main__':
    main()
