import torch
import random
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from dataset import ArgoverseNeighborDataset
from model   import ContextualSocialLSTM

def compute_ade_fde(preds: torch.Tensor,
                    gts:   torch.Tensor):
    # preds, gts: [B × L × 2]
    # Euclidean distance at each horizon:
    dists = torch.norm(preds - gts, dim=-1)  # [B × L]
    # ADE = average over batch and time:
    ade = dists.mean()
    # FDE = last time-step error averaged over batch:
    fde = dists[:, -1].mean()
    return ade.item(), fde.item(), dists  # return full dists if you want per‐horizon


def visualize_predictions(model, val_loader, mu_xy, sigma_xy, device, num_examples=5):
    model.eval()
    # grab a handful of batches until we have num_examples
    collected = 0
    figs = []
    with torch.no_grad():
        for hist_e, hist_t, hist_o, mask_o, ctx, intent, fut in val_loader:
            hist_e = hist_e.to(device)
            hist_t = hist_t.to(device)
            hist_o = hist_o.to(device)
            mask_o = mask_o.to(device)
            ctx    = ctx.to(device)
            intent = intent.to(device)
            fut    = fut.to(device)

            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx, intent)  # [B×L×2]
            # only pred_len=1 case? if longer, you can plot whole curve
            # here we assume pred_len==1
            # denormalize both
            # hist_t[:,:,0:2] + fut[:,0:2], pred_seq[:,:,0:2]
            hist_xy = hist_t[:, :, :2] * sigma_xy + mu_xy    # [B×T×2]
            fut_xy  = fut * sigma_xy + mu_xy    # [B×1×2]
            pred_xy = pred_seq * sigma_xy + mu_xy            # [B×1×2]

            B = hist_xy.size(0)
            for b in range(B):
                figs.append((hist_xy[b].cpu().numpy(),
                             fut_xy[b,0].cpu().numpy(),
                             pred_xy[b,0].cpu().numpy()))
                collected += 1
                if collected >= num_examples:
                    break
            if collected >= num_examples:
                break

    # now plot
    for i, (hist, fut, pred) in enumerate(figs):
        plt.figure(figsize=(4,4))
        # history
        plt.plot(hist[:,0], hist[:,1], '-o', color='gray', label='history')
        # true future
        plt.scatter(fut[0], fut[1], color='green', s=100, marker='*', label='ground truth')
        # predicted
        plt.scatter(pred[0], pred[1], color='red', s=100, marker='X', label='prediction')
        plt.title(f"Val Sample #{i+1}")
        plt.xlabel("Δx (m)"); plt.ylabel("Δy (m)")
        plt.axis('equal')
        plt.legend()
        plt.show()



def train(
    ego_csv: str,
    social_csv: str,
    contextual_npy: str,
    seq_len: int,
    pred_len: int,
    max_neighbors: int,
    neighbor_radius: float,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    use_delta_yaw: bool,
    use_context: bool,
    use_intention: bool
):

    # 1) Dataset
    ds = ArgoverseNeighborDataset(
        ego_path        = ego_csv,
        social_path     = social_csv,
        contextual_path = contextual_npy,
        seq_len         = seq_len,
        pred_len        = pred_len,
        max_neighbors   = max_neighbors,
        use_delta_yaw   = use_delta_yaw,
        use_context     = use_context,
        use_intention   = use_intention,
        neighbor_radius = neighbor_radius
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
        intent_dim    = 3 if use_intention else 0,
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

        loader = train_loader
        pbar   = tqdm(loader, desc=f"Epoch {ep:>2}", unit="batch")
        for i, (hist_e, hist_t, hist_o, mask_o, ctx,intent, fut) in enumerate(pbar):
            t0 = time.perf_counter()

            # 1) measure host→device copy
            t_before_copy = time.perf_counter()
            hist_e = hist_e.to(device, non_blocking=True)
            hist_t = hist_t.to(device, non_blocking=True)
            hist_o = hist_o.to(device, non_blocking=True)
            mask_o = mask_o.to(device, non_blocking=True)
            ctx    = ctx.to(device,    non_blocking=True)
            intent = intent.to(device, non_blocking=True)
            fut    = fut.to(device,    non_blocking=True)
            t_after_copy = time.perf_counter()

            # 2) measure forward/backward
            opt.zero_grad()
            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx, intent)
            loss     = loss_fn(pred_seq, fut)
            loss.backward()
            opt.step()
            t_after_step = time.perf_counter()

            total_loss += loss.item()

            # 3) update bar with timings
            if i % 50 == 0:
                copy_time   = (t_after_copy - t_before_copy)
                fwd_time    = (t_after_step - t_after_copy)
                pbar.set_postfix({
                    "copy_ms": f"{copy_time*1e3:5.1f}",
                    "fwd_ms":  f"{fwd_time*1e3:5.1f}"
                })

        avg_loss = total_loss / len(train_loader)

       # 7) Validation (multi‐horizon ADE)
    model.eval()
    sum_dists  = torch.zeros(model.pred_len, device=device)
    sum_ade_m, sum_fde_m, cnt = 0.0, 0.0,0
    total_seqs = 0
    with torch.no_grad():
        for hist_e, hist_t, hist_o, mask_o, ctx, intent, fut in val_loader:
            # move to device
            hist_e, hist_t, hist_o = [x.to(device, non_blocking=True) for x in (hist_e, hist_t, hist_o)]
            mask_o                  = mask_o.to(device,    non_blocking=True)
            ctx, intent, fut        = [x.to(device,    non_blocking=True) for x in (ctx, intent, fut)]

            # forward
            pred_seq = model(hist_e, hist_t, hist_o, mask_o, ctx, intent)  # [B × pred_len × 2]

            # de-normalize
            pred_m = pred_seq * sigma_xy + mu_xy   # [B × pred_len × 2]
            gt_m   = fut      * sigma_xy + mu_xy   # [B × pred_len × 2]

            # L2 error per horizon
            dists = torch.norm(pred_m - gt_m, dim=-1)  # [B × pred_len]
            # overall ADE / FDE accumulation
            ade_m, fde_m = compute_ade_fde(pred_m, gt_m)
            sum_ade_m += ade_m * fut.size(0)
            sum_fde_m += fde_m * fut.size(0)
            cnt      += fut.size(0)


            sum_dists  += dists.sum(dim=0)   # accumulate over batch
            total_seqs += dists.size(0)

        dists    = torch.norm(pred_m - gt_m, dim=-1)    # [B×pred_len]
        ade_per_t= dists.mean(dim=0)                    # [pred_len]
        fde_per_t= dists[:, -1]                         # [B]
        print()
        # compute ADE per horizon
        ade_per_t = sum_dists / total_seqs    # [pred_len]
        for h, ade_h in enumerate(ade_per_t, start=1):
            print(f"  → Horizon {h}: ADE = {ade_h:.4f} m")
        print(f"  Final‐step FDE: {fde_per_t.mean():.4f} m")
        print()
    val_ade_m = sum_ade_m / cnt
    val_fde_m = sum_fde_m / cnt
    overall_ade = ade_per_t.mean().item()
    final_fde   = ade_per_t[-1].item()
    print(f"  Overall ADE = {overall_ade:.4f} m | Final-step FDE = {final_fde:.4f} m")
    print(f"Epoch {ep}: Train L {avg_loss:.4f} | Val ADE {val_ade_m:.4f} m | FDE {val_fde_m:.4f} m")

    # snapshot best
    if overall_ade < best_val:
        best_val = overall_ade
        torch.save(model.state_dict(), "best_contextual_social_lstm.pth")
    # === at the bottom of your train() function, after saving the model ===
    # assuming you still have: 
    #   model, val_loader, mu_xy, sigma_xy, device
    print("\n📊 Visualization of a few validation predictions:")
    visualize_predictions(model, val_loader, mu_xy, sigma_xy, device, num_examples=5)

    return best_val

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ego_csv",      default="data/processed/ego_vehicle_with_intention.csv")
    p.add_argument("--social_csv",   default="data/processed/social_vehicles_relative.csv")
    p.add_argument("--contextual_npy",  default="data/processed/contextual_features_merged.npy")
    p.add_argument("--seq_len",  type=int,   default=30)
    p.add_argument("--pred_len", type=int,   default=30)
    p.add_argument("--max_neighbors", type=int, default=10)
    p.add_argument("--neighbor_radius", type=float, default=10.0)
    p.add_argument("--batch",    type=int,   default=64)
    p.add_argument("--hidden",   type=int,   default=128)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--epochs",   type=int,   default=20)
    p.add_argument("--use_delta_yaw", action="store_true",
                   help="Include delta_yaw as an extra feature channel"),
    p.add_argument("--no_context",   action="store_true", help="disable contextual features")
    p.add_argument("--use_intention", action="store_true",
                   help="add 3-way turn/straight intention to context")
    args = p.parse_args()

    # Pull all args into a dict, but pop off `use_delta_yaw`
    params = {
        "ego_csv":        args.ego_csv,
        "social_csv":     args.social_csv,
        "contextual_npy": args.contextual_npy,
        "seq_len":        args.seq_len,
        "pred_len":       args.pred_len,
        "max_neighbors":  args.max_neighbors,
        "neighbor_radius": args.neighbor_radius,
        "batch_size":     args.batch,
        "hidden_dim":     args.hidden,
        "lr":             args.lr,
        "epochs":         args.epochs,
        "use_delta_yaw":  args.use_delta_yaw,
        "use_context":    not args.no_context,
        "use_intention":  args.use_intention
    }

    train(**params)

if __name__ == "__main__":
    main()
