# merge_context_features.py
import numpy as np

# === Correct files to use ===
lane_features_path = "data/processed/contextual_features_enhanced.npy"  # Enhanced lanes
object_features_path = "data/processed/contextual_objects.npy"   # Map object context
output_path = "data/processed/contextual_features_merged.npy"

# === Load
lane_context = np.load(lane_features_path)
object_context = np.load(object_features_path)

print("Lane features shape:", lane_context.shape)
print("Object features shape:", object_context.shape)

# === Align
min_len = min(len(lane_context), len(object_context))
lane_context = lane_context[:min_len]
object_context = object_context[:min_len]

# === Merge
final_context = np.concatenate([lane_context, object_context], axis=1)
print("✅ Final merged context shape:", final_context.shape)  # Should be (min_len, 15)

# === Save
np.save(output_path, final_context)
print(f"✅ Saved merged contextual features to {output_path}")
