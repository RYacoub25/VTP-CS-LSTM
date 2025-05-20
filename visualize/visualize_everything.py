import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Add parent directory of visualize folder (which contains CS_LSTM) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CS_LSTM.dataset import ArgoverseNeighborDataset
from CS_LSTM.model import ContextualSocialLSTM

# Map object colors
TARGET_CLASSES = {
    "STOP_SIGN": "red",
    "SIGN": "orange",
    "TRAFFIC_LIGHT_TRAILER": "purple",
    "CONSTRUCTION_CONE": "blue",
    "CONSTRUCTION_BARREL": "brown",
    "MESSAGE_BOARD_TRAILER": "pink",
    "MOBILE_PEDESTRIAN_SIGN": "cyan"
}

def social_to_global(social_df_ts, ego_x, ego_y):
    social_global = social_df_ts.copy()
    social_global["x"] += ego_x
    social_global["y"] += ego_y
    return social_global

def plot_icon(ax, x, y, img_path, zoom=0.05):
    img = mpimg.imread(img_path)
    im = OffsetImage(img, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False)
    ax.add_artist(ab)

def plot_scene_with_predictions(
    ts, ego_df, social_df, lane_df, map_df,
    fut_relative, pred_relative,
    history_relative, radius
):
    ego_at_ts = ego_df[ego_df["timestamp_ns"] == ts]
    if ego_at_ts.empty:
        print(f"No ego vehicle data at timestamp {ts}")
        return
    ego_x, ego_y = ego_at_ts.iloc[0]["x"], ego_at_ts.iloc[0]["y"]
    ego_pos = np.array([ego_x, ego_y])

    # social positions are relative; convert to global
    social_at_ts = social_df[social_df["timestamp_ns"] == ts].copy()
    social_at_ts["x"] += ego_x
    social_at_ts["y"] += ego_y

    fig, ax = plt.subplots(figsize=(12,12))

    # Plot lanes
    for _, row in lane_df.iterrows():
        try:
            left = eval(row["left_lane_boundary"])
            right = eval(row["right_lane_boundary"])
            if isinstance(left, list) and len(left) > 1:
                lx = [p["x"] for p in left]
                ly = [p["y"] for p in left]
                ax.plot(lx, ly, c='green', linewidth=2, alpha=0.4)
            if isinstance(right, list) and len(right) > 1:
                rx = [p["x"] for p in right]
                ry = [p["y"] for p in right]
                ax.plot(rx, ry, c='darkorange', linewidth=2, alpha=0.4)
        except Exception:
            continue

    # Plot map objects
    map_ts = map_df[map_df["timestamp_ns"] == ts]
    TARGET_CLASSES = {
        "STOP_SIGN": "red",
        "SIGN": "orange",
        "TRAFFIC_LIGHT_TRAILER": "purple",
        "CONSTRUCTION_CONE": "blue",
        "CONSTRUCTION_BARREL": "brown",
        "MESSAGE_BOARD_TRAILER": "pink",
        "MOBILE_PEDESTRIAN_SIGN": "cyan"
    }
    for cls, color in TARGET_CLASSES.items():
        cls_objs = map_ts[map_ts["category"] == cls]
        if not cls_objs.empty:
            ax.scatter(
                cls_objs["x"], cls_objs["y"],
                c=color, label=cls.replace("_", " ").title(), alpha=0.8
            )

    # Plot social vehicles
    ax.scatter(
        social_at_ts["x"], social_at_ts["y"],
        c='blue', label="Social Vehicles", alpha=0.7, zorder=3
    )

    # Plot ego vehicle with car icon
    plot_icon(ax, ego_x, ego_y, "data/car.png", zoom=0.05)

    # Plot target vehicle as purple star at last history point
    last_rel = history_relative[-1]
    target_global = ego_pos + last_rel
    ax.scatter(
        target_global[0], target_global[1],
        c='purple', s=200, marker='*',
        edgecolors='black', linewidths=1.5,
        zorder=5, label="Target Vehicle"
    )

    # Build trajectories
    # history_relative: [T×2], fut_relative & pred_relative: [L×2]
    fut_abs_rel  = last_rel + np.cumsum(fut_relative, axis=0)
    pred_abs_rel = last_rel + np.cumsum(pred_relative, axis=0)

    history_global = ego_pos + history_relative
    fut_global     = ego_pos + fut_abs_rel
    pred_global    = ego_pos + pred_abs_rel

    # Plot target history
    ax.plot(
        history_global[:,0], history_global[:,1],
        'o-', c='gray', lw=2, ms=6, zorder=4, label='Target History'
    )

    # Plot ground truth future
    ax.plot(
        fut_global[:,0], fut_global[:,1],
        's-', c='green', lw=2, ms=6, zorder=4, label='Ground Truth'
    )

    # Plot predicted future
    ax.plot(
        pred_global[:,0], pred_global[:,1],
        'x-', c='red', lw=2, ms=8, zorder=4, label='Prediction'
    )

    ax.set_aspect('equal', 'box')
    ax.set_xlim(ego_x - radius, ego_x + radius)
    ax.set_ylim(ego_y - radius, ego_y + radius)
    ax.set_title(f"Scene at Timestamp: {ts}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True)
    ax.legend(loc='upper left')
    plt.show()



def visualize_batch(model, val_loader, ego_df, social_df, lane_df, map_df,
                    mu_xy, sigma_xy, device, num_examples=5, radius=30):
    model.eval()
    shown = 0

    with torch.no_grad():
        for batch in val_loader:
            # unpack (now assuming __getitem__ returns tid and ts at end)
            hist_e, hist_t, hist_o, mask_o, ctx, intent, fut, tid, ts = batch

            # move to device
            hist_t = hist_t.to(device)   # [B×T×F]
            fut    = fut.to(device)      # [B×L×2]
            pred   = model(hist_e.to(device),
                           hist_t, hist_o.to(device),
                           mask_o.to(device),
                           ctx.to(device),
                           intent.to(device))   # [B×L×2]

            # denormalize only x,y channels
            mu = mu_xy.to(device)
            sd = sigma_xy.to(device)

            # history_relative: target’s x,y history
            history_rel = hist_t[:,:, :2] * sd + mu    # [B×T×2]
            # future & pred are relative displacements
            fut_rel  = fut  * sd                       # [B×L×2]
            pred_rel = pred * sd                       # [B×L×2]

            B = history_rel.size(0)
            for b in range(B):
                h = history_rel[b].cpu().numpy()    # [T×2]
                f = fut_rel[b].cpu().numpy()        # [L×2]
                p = pred_rel[b].cpu().numpy()       # [L×2]
                ts_i = int(ts[b].item())
                plot_scene_with_predictions(
                    ts=ts_i,
                    ego_df=ego_df,
                    social_df=social_df,
                    lane_df=lane_df,
                    map_df=map_df,
                    history_relative=h,
                    fut_relative=f,
                    pred_relative=p,
                    radius=radius
                )
                shown += 1
                if shown >= num_examples:
                    return

    print("Done visualizing.")
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and model
    checkpoint = torch.load("best_contextual_social_lstm.pth", map_location=device)
    config = checkpoint['config']
    print(config)
    model = ContextualSocialLSTM(
        input_dim=config['input_dim'],
        context_dim=config['context_dim'],
        intent_dim=config['intent_dim'],
        hidden_dim=config['hidden_dim'],
        pred_len=50,
        use_delta_yaw=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load dataframes
    ego_df = pd.read_csv("data/processed/ego_vehicle_with_intention.csv")
    social_df = pd.read_csv("data/processed/social_vehicles_relative.csv")
    lane_df = pd.read_csv("data/processed/constant_features.csv")
    map_df = pd.read_csv("data/processed/map_objects.csv")

    # Dataset and DataLoader for validation
    val_dataset = ArgoverseNeighborDataset(
        ego_path="data/processed/ego_vehicle_with_intention.csv",
        social_path="data/processed/social_vehicles_relative.csv",
        contextual_path="data/processed/contextual_features_merged.npy",
        seq_len=50,
        pred_len=50,
        target_radius=20,
        max_neighbors=None,
        use_delta_yaw=True,
        use_context=True,
        use_intention=True,
        neighbor_radius=10
    )
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    mu_xy = torch.tensor(val_dataset.feat_mean[0, :2], dtype=torch.float32)
    sigma_xy = torch.tensor(val_dataset.feat_std[0, :2], dtype=torch.float32)

    # Visualize a few samples with scene
    num_examples = 1
    with torch.no_grad():
        for b,( hist_e, hist_t, hist_o, mask_o, ctx, intent, fut,tid,ts) in enumerate(val_loader):
            hist_e = hist_e.to(device)
            hist_t = hist_t.to(device)
            hist_o = hist_o.to(device)
            mask_o = mask_o.to(device)
            ctx = ctx.to(device)
            intent = intent.to(device)
            fut = fut.to(device)

            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx, intent)  # [B×L×2]
            mu_xy = mu_xy.to(device)
            sigma_xy = sigma_xy.to(device)
            hist_xy = hist_t[:, :, :2] * sigma_xy + mu_xy       # [B×T×2]
            fut_disp = fut * sigma_xy                            # [B×L×2]
            pred_disp = pred_seq * sigma_xy      
            

            visualize_batch(
                model, val_loader,
                ego_df, social_df, lane_df, map_df,
                mu_xy, sigma_xy, device,
                num_examples=1,
                radius=30
            )

            # for i in range(hist_xy.size(0)):
            #     print(f"\nSample {b*val_loader.batch_size + i}")
            #     print(f"Timestamp: {ts[i].item() if hasattr(ts[i], 'item') else ts[i]}")
            #     print(f"Target track UUID: {tid[i]}")
            #     print(f"Ego last position (normalized + denorm): {hist_e[i,-1,:2].cpu().numpy() * sigma_xy.cpu().numpy() + mu_xy.cpu().numpy()}")
            #     print(f"Target last history pos: {hist_xy[i,-1].cpu().numpy()}")
            #     print(f"Target future displacement shape: {fut_disp[i].shape}")
            #     print(f"Target predicted displacement shape: {pred_disp[i].shape}")
            #     print(f"First 3 future displacements (ground truth): {fut_disp[i][:3].cpu().numpy()}")
            #     print(f"First 3 predicted displacements: {pred_disp[i][:3].cpu().numpy()}")

            #     # Optional: print social vehicles positions at this timestamp
            #     soc_at_ts = social_df[social_df["timestamp_ns"] == ts[i].item()]
            #     ego_at_ts = ego_df[ego_df["timestamp_ns"] == ts[i].item()]
            #     print(f"Number of social vehicles at ts: {len(soc_at_ts)}")
            #     print(f"Ego vehicle position at ts: ({ego_at_ts.iloc[0]['x']}, {ego_at_ts.iloc[0]['y']})")

            # # Add break if you want to limit output
            # if b > 2:
            #     break
            #             # [B×L×2]

            # B = hist_xy.size(0)
            # for b in range(B):
            #     history_global = hist_xy[b].cpu().numpy()
            #     fut_relative = fut_disp[b].cpu().numpy()
            #     pred_relative = pred_disp[b].cpu().numpy()

            #     ts_idx = b if b < len(val_dataset.timestamps) else 0
            #     ts = val_dataset.timestamps[ts_idx]
            #     target_pos = history_global[-1]  # last point in history as the current target position

            #     plot_scene_with_predictions(
            #         ts, ego_df, social_df, lane_df, map_df,
            #         fut_relative, pred_relative, history_global,
            #         radius=30,

            #     )
            #     plot_scene_with_predictions(
            #         ts, ego_df, social_df, lane_df, map_df,
            #         fut_relative, pred_relative, history_global,
            #         radius=15,

            #     )

            #     num_examples -= 1
            #     if num_examples <= 0:
            #         return

if __name__ == "__main__":
    main()
