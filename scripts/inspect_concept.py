#!/usr/bin/env python3
import numpy as np

def main():
    path = "data/processed/contextual_features_merged.npy"  # adjust if needed
    ctx = np.load(path)
    B, C = ctx.shape
    print(f"Loaded context array: shape = {ctx.shape}, dtype = {ctx.dtype}\n")

    # print a summary (min/max/mean) for each feature column
    print("  idx │   min    │   max    │   mean")
    print("─────┼─────────┼─────────┼─────────")
    for i in range(C):
        col = ctx[:, i]
        print(f"{i:4d} │ {col.min():7.3f} │ {col.max():7.3f} │ {col.mean():7.3f}")
    print()

    # show the first sample’s feature vector
    print("First row (sample 0):")
    print(ctx[0])

if __name__ == "__main__":
    main()
