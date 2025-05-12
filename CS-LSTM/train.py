import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ArgoverseNeighborDataset
from model   import ContextualSocialLSTM

def compute_ade_fde(preds: torch.Tensor,
                    gts:   torch.Tensor) -> (float, float):
    if preds.dim() == 2:
        preds = preds.unsqueeze(1)
        gts   = gts.unsqueeze(1)
    dists = torch.norm(preds - gts, dim=-1)
    ade   = dists.mean()
    fde   = dists[:, -1].mean()
    return ade.item(), fde.item()

def train(
    ego_csv:        str,
    social_csv:     str,
    contextual_npy: str,
    seq_len:        int = 30,
    pred_len:       int = 1,
    max_neighbors:  int = 20,
    batch_size:     int = 64,
    hidden_dim:     int = 128,
    lr:             float = 1e-3,
    epochs:         int = 5,
    use_delta_yaw: bool = False,
):
    # 1) Build dataset
    ds = ArgoverseNeighborDataset(
        ego_path        = ego_csv,
        social_path     = social_csv,
        contextual_path = contextual_npy,
        seq_len         = seq_len,
        pred_len        = pred_len,
        max_neighbors   = max_neighbors,
        use_delta_yaw   = use_delta_yaw,
    )

    # 2) Setup device & de-normalization stats
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu_xy     = torch.tensor(ds.feat_mean[0, :2], device=device)   # [2]
    sigma_xy  = torch.tensor(ds.feat_std[0,  :2], device=device)   # [2]
    
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())

    # 3) Print some frame intervals
    dts_ns = [ds.timestamps[i+1] - ds.timestamps[i] for i in range(5)]
    print("First 5 frame intervals (s):",
          [dt/1e9 for dt in dts_ns])

    # 4) Split train/val
    n_val = int(0.15 * len(ds))
    n_trn = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_trn, n_val])

    # # 5) Find & visualize the “crowded” sample
    # neighbor_counts = [int(mask_o[-1].sum()) for _,_,_,mask_o,_,_ in ds]
    # best_idx = int(np.argmax(neighbor_counts))
    # print(f"Most crowded sample index: {best_idx} ({neighbor_counts[best_idx]} neighbors)")

    # hist_e, hist_t, hist_o, mask_o, ctx, fut = ds[best_idx]
    # # de-normalize neighbor coords (meters)
    # coords_norm = hist_o[-1][mask_o[-1].astype(bool), :2]
    # coords_m    = coords_norm * sigma_xy.cpu().numpy() + mu_xy.cpu().numpy()

    # # jitter & plot
    # jitter = np.random.normal(scale=0.2, size=coords_m.shape)
    # coords_j = coords_m + jitter

    # plt.figure()
    # plt.scatter(coords_j[:,0], coords_j[:,1], alpha=0.7, label="neighbors (m)")
    # plt.scatter(0, 0, c="red", s=100, label="ego")
    # plt.title(f"Jittered Neighbors for sample {best_idx}")
    # plt.xlabel("Δx (m)"); plt.ylabel("Δy (m)")
    # plt.legend(); plt.axis("equal"); plt.show()

    # # de-normalize history & future and plot
    # hist_xy_m = hist_t[:, :2] * sigma_xy.cpu().numpy() + mu_xy.cpu().numpy()
    # fut_m     = fut * sigma_xy.cpu().numpy() + mu_xy.cpu().numpy()

    # plt.figure()
    # plt.plot(hist_xy_m[:,0], hist_xy_m[:,1], "-o", label="history (m)")
    # plt.scatter(fut_m[0], fut_m[1], c="red", label="true future (m)")
    # plt.title(f"Relative Trajectory for sample {best_idx}")
    # plt.xlabel("Δx (m)"); plt.ylabel("Δy (m)")
    # plt.axis("equal"); plt.legend(); plt.show()

    # 6) Build DataLoaders
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, persistent_workers=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=2, persistent_workers=True, pin_memory=True)


    # 7) Build model, optimizer, loss
    model   = ContextualSocialLSTM(
        input_dim     = len(ds.features),
        context_dim   = ds.context.shape[1],
        hidden_dim    = hidden_dim,
        max_neighbors = max_neighbors,
        pred_len      = pred_len
    ).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 8) Training & validation loop
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for hist_e, hist_t, hist_o, mask_o, ctx, fut in tqdm(
            train_loader, desc=f"Epoch {ep:>2}", unit="batch"
        ):
            hist_e = hist_e.to(device)
            hist_t = hist_t.to(device)
            hist_o = hist_o.to(device)
            mask_o = mask_o.to(device)
            ctx    = ctx.to(device)
            fut    = fut.to(device)

            opt.zero_grad()
            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx)  # norm
            gt_seq   = fut.unsqueeze(1)                            # norm

            loss = loss_fn(pred_seq, gt_seq)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # validation in meters
        model.eval()
        sum_ade_m, sum_fde_m, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for hist_e, hist_t, hist_o, mask_o, ctx, fut in val_loader:
                hist_e = hist_e.to(device)
                hist_t = hist_t.to(device)
                hist_o = hist_o.to(device)
                mask_o = mask_o.to(device)
                ctx    = ctx.to(device)
                fut    = fut.to(device)

                pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx)  # norm
                gt_seq   = fut.unsqueeze(1)                            # norm

                # de-normalize into meters
                pred_m = pred_seq * sigma_xy + mu_xy   # [B×L×2] in m
                gt_m   = gt_seq   * sigma_xy + mu_xy

                ade_m, fde_m = compute_ade_fde(pred_m, gt_m)
                sum_ade_m += ade_m * fut.size(0)
                sum_fde_m += fde_m * fut.size(0)
                cnt      += fut.size(0)

        val_ade_m = sum_ade_m / cnt
        val_fde_m = sum_fde_m / cnt
        print(f"Epoch {ep}: Train Loss {avg_loss:.4f} | Val ADE {val_ade_m:.4f} m | Val FDE {val_fde_m:.4f} m")

    # 9) Save final model
    torch.save(model.state_dict(), "contextual_social_lstm_argo.pth")
    print("✅ Model saved to contextual_social_lstm_argo.pth")
    return val_ade_m

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ego",      default="data/processed/ego_vehicle_with_intention.csv")
    p.add_argument("--social",   default="data/processed/social_vehicles_relative.csv")
    p.add_argument("--context",  default="data/processed/contextual_features_merged.npy")
    p.add_argument("--seq_len",  type=int,   default=30)
    p.add_argument("--pred_len", type=int,   default=1)
    p.add_argument("--max_nb",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--use_delta_yaw", action="store_true",help="Include delta_yaw as an extra feature channel")
    args=p.parse_args()

    #Run ablation: without Δ-yaw, then with Δ-yaw
    ade_yes = train(
        ego_csv        = args.ego,
        social_csv     = args.social,
        contextual_npy = args.context,
        seq_len        = args.seq_len,
        pred_len       = args.pred_len,
        max_neighbors  = args.max_nb,
        batch_size     = args.batch,
        hidden_dim     = args.hidden,
        lr             = args.lr,
        epochs         = args.epochs,
        use_delta_yaw  = True
    )
    ade_no = train(
        ego_csv        = args.ego,
        social_csv     = args.social,
        contextual_npy = args.context,
        seq_len        = args.seq_len,
        pred_len       = args.pred_len,
        max_neighbors  = args.max_nb,
        batch_size     = args.batch,
        hidden_dim     = args.hidden,
        lr             = args.lr,
        epochs         = args.epochs,
        use_delta_yaw  = False
    )

    # Plot the comparison bar chart
    import matplotlib.pyplot as plt
    plt.figure()
    plt.bar(
        ['+ Δ-yaw', 'no Δ-yaw'],
        [ade_yes, ade_no],
        color=['skyblue', 'orange'],
        width=0.4
    )
    plt.ylabel('Best Val ADE (m)')
    plt.title('Effect of Δ-yaw on Single-Step ADE')
    plt.show()