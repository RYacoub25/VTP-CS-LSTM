import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import ArgoverseNeighborDataset
from model   import ContextualSocialLSTM

def compute_ade_fde(preds: torch.Tensor, gts: torch.Tensor) -> (float, float):
    if preds.dim() == 2:
        preds = preds.unsqueeze(1)
        gts   = gts.unsqueeze(1)
    dists = torch.norm(preds - gts, dim=-1)
    return float(dists.mean()), float(dists[:, -1].mean())

def train(
    ego_csv: str,
    social_csv: str,
    contextual_npy: str,
    seq_len: int,
    pred_len: int,
    max_neighbors: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    use_delta_yaw: bool,
    use_context: bool
):
    # 1) Dataset
    ds = ArgoverseNeighborDataset(
        ego_path        = ego_csv,
        social_path     = social_csv,
        contextual_path = contextual_npy,
        seq_len         = seq_len,
        pred_len        = pred_len,
        max_neighbors   = max_neighbors,
        use_delta_yaw   = True,
        use_context     = True,
    )

    # 2) Device & de-normal stats
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu_xy     = torch.tensor(ds.feat_mean[0,:2], device=device)
    sigma_xy  = torch.tensor(ds.feat_std[0,:2],  device=device)
    print("Using device:", device)

    # 3) Split
    n_val = int(0.15 * len(ds))
    n_trn = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_trn, n_val])

    # 4) Loaders (parallel & pinned)
    train_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
   #     prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
   #     prefetch_factor=2
    )

    # 5) Model, opt, loss
    model = ContextualSocialLSTM(
        input_dim     = len(ds.features),
        context_dim   = ds.context.shape[1],
        hidden_dim    = hidden_dim,
        max_neighbors = max_neighbors,
        pred_len      = pred_len
    ).to(device)
    opt     = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 6) Training loop
    best_val = float('inf')
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for hist_e, hist_t, hist_o, mask_o, ctx, fut in tqdm(
            train_loader, desc=f"Epoch {ep:>2}", unit="batch"
        ):
            # async copy to GPU
            hist_e = hist_e.to(device, non_blocking=True)
            hist_t = hist_t.to(device, non_blocking=True)
            hist_o = hist_o.to(device, non_blocking=True)
            mask_o = mask_o.to(device, non_blocking=True)
            ctx    = ctx.to(device,    non_blocking=True)
            if not use_context:
                # disable all contextual features
                ctx = torch.zeros_like(ctx)
            fut    = fut.to(device,    non_blocking=True)

            opt.zero_grad()
            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx)
            loss     = loss_fn(pred_seq, fut.unsqueeze(1))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 7) Validation (in meters)
        model.eval()
        sum_ade_m, sum_fde_m, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for hist_e, hist_t, hist_o, mask_o, ctx, fut in val_loader:
                hist_e = hist_e.to(device, non_blocking=True)
                hist_t = hist_t.to(device, non_blocking=True)
                hist_o = hist_o.to(device, non_blocking=True)
                mask_o = mask_o.to(device, non_blocking=True)
                ctx    = ctx.to(device, non_blocking=True)
                fut    = fut.to(device, non_blocking=True)

                pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx)
                gt_seq   = fut.unsqueeze(1)

                # de-normalize and compute ADE/FDE
                pred_m = pred_seq * sigma_xy + mu_xy
                gt_m   = gt_seq   * sigma_xy + mu_xy
                ade_m, fde_m = compute_ade_fde(pred_m, gt_m)

                sum_ade_m += ade_m * fut.size(0)
                sum_fde_m += fde_m * fut.size(0)
                cnt      += fut.size(0)

        val_ade_m = sum_ade_m / cnt
        val_fde_m = sum_fde_m / cnt
        print(f"Epoch {ep}: Train L {avg_loss:.4f} | Val ADE {val_ade_m:.4f} m | FDE {val_fde_m:.4f} m")

        if val_ade_m < best_val:
            best_val = val_ade_m
            torch.save(model.state_dict(), "best_contextual_social_lstm.pth")

    return best_val

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ego_csv",      default="data/processed/ego_vehicle_with_intention.csv")
    p.add_argument("--social_csv",   default="data/processed/social_vehicles_relative.csv")
    p.add_argument("--contextual_npy",  default="data/processed/contextual_features_merged.npy")
    p.add_argument("--seq_len",  type=int,   default=30)
    p.add_argument("--pred_len", type=int,   default=1)
    p.add_argument("--max_neighbors",   type=int,   default=20)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--use_delta_yaw", action="store_true",
                   help="Include delta_yaw as an extra feature channel"),
    p.add_argument("--no_context",   action="store_true", help="disable contextual features")
    args = p.parse_args()

    # Pull all args into a dict, but pop off `use_delta_yaw`
    params = {
        "ego_csv":        args.ego_csv,
        "social_csv":     args.social_csv,
        "contextual_npy": args.contextual_npy,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "max_neighbors":  args.max_neighbors,
        "batch_size":     args.batch,
        "hidden_dim":     args.hidden,
        "lr":             args.lr,
        "epochs":         args.epochs,
        "use_delta_yaw":  True,
        "use_context":    True
    }

    # Run +Δ-yaw then –Δ-yaw
    best_without    = train(**params, use_context=False)
    best_with = train(**params, use_context=True)

#    8) Bar chart
    plt.figure()
    plt.bar(['+ context', 'no context'],
            [best_with, best_without],
            width=0.5)
    plt.ylabel('Best Val ADE (m)')
    plt.title('Context Ablation')
    plt.show()

if __name__ == "__main__":
    main()
