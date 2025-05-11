import pandas as pd
import os
from tqdm import tqdm
from config import SENSOR_DIR

# Directory where raw data is stored
output_path = "data/processed/map_objects.csv"

# Classes related to map and environment context
map_classes = [
    "STOP_SIGN", "SIGN", "CONSTRUCTION_CONE", "CONSTRUCTION_BARREL",
    "MOBILE_PEDESTRIAN_SIGN", "TRAFFIC_LIGHT_TRAILER", "MESSAGE_BOARD_TRAILER"
]

def extract_from_annotation(path):
    try:
        df = pd.read_feather(path)
        filtered = df[df['category'].isin(map_classes)]
        return filtered
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
        return pd.DataFrame()

def extract_map_objects():
    all_filtered = []

    print(f"🔍 Searching for 'annotations.feather' files in {SENSOR_DIR}...\n")
    for root, _, files in tqdm(os.walk(SENSOR_DIR)):
        for file in files:
            if file == 'annotations.feather':
                path = os.path.join(root, file)
                print(f"📂 Annotation file: {path}")
                filtered = extract_from_annotation(path)
                if not filtered.empty:
                    all_filtered.append(filtered)

    if not all_filtered:
        print("⚠️ No relevant map-related objects found.")
        return

    result_df = pd.concat(all_filtered, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved {len(result_df)} entries to {output_path}")

if __name__ == "__main__":
    extract_map_objects()
