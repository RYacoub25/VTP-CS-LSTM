import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

class ArgoverseNeighborDataset(Dataset):
    def __init__(self,
                 ego_path: str,
                 social_path: str,
                 contextual_path: str,
                 seq_len: int = 30,
                 pred_len: int = 1,
                 max_neighbors: int = 10,
                 use_delta_yaw: bool = False,
                 features=None):
        if features is None:
            features = ['x','y','z','vx','vy','vz','ax','ay','az']
            if use_delta_yaw:
                features.append('delta_yaw')
        self.seq_len       = seq_len
        self.pred_len      = pred_len
        self.max_neighbors = max_neighbors
        self.use_delta_yaw = use_delta_yaw
        self.features      = features

        # for diagnostics
        self._neighbor_counts = []

        # 1) load ego CSV
        self.ego_df = pd.read_csv(ego_path)
        # — ego has yaw but no delta_yaw: compute it if requested
        if use_delta_yaw:
            self.ego_df = self.ego_df.sort_values('timestamp_ns')
            self.ego_df['delta_yaw'] = self.ego_df['yaw'].diff().fillna(0.0)

        # 2) load social CSV
        self.social_df = pd.read_csv(social_path)
        # — social already has delta_yaw in your file; only compute if missing
        if use_delta_yaw and 'delta_yaw' not in self.social_df.columns:
            self.social_df = self.social_df.sort_values(['track_uuid','timestamp_ns'])
            self.social_df['delta_yaw'] = (
                self.social_df
                    .groupby('track_uuid')['yaw']
                    .diff()
                    .fillna(0.0)
            )

        self.context   = np.load(contextual_path)




        # 2) group by timestamp
        self.ego_group    = self.ego_df.groupby('timestamp_ns')
        self.social_group = self.social_df.groupby('timestamp_ns')

        # 3) intersect timestamps
        ego_times    = set(self.ego_group.groups.keys())
        social_times = set(self.social_group.groups.keys())
        self.timestamps = sorted(ego_times & social_times)


        
        # 4) cache per-timestamp arrays
        self.social_data = {}
        for ts, df_ts in self.social_group:
            if ts not in social_times: continue
            ids     = df_ts['track_uuid'].values
            feats   = df_ts[self.features].values
            idx_map = {uid: i for i, uid in enumerate(ids)}
            self.social_data[ts] = (ids, feats, idx_map)

        self.ego_data = {
            ts: df_ts[self.features].values[0]
            for ts, df_ts in self.ego_group
            if ts in ego_times
        }

        # 5) compute feature-means/stds
        all_feats = np.vstack([feats for _, feats, _ in self.social_data.values()])
        self.feat_mean = all_feats.mean(axis=0, keepdims=True)
        self.feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-6

        # 6) build samples
        self.samples = self._prepare_samples()
        for i, sample in enumerate(self.samples):
            self.samples[i] = {
                'hist_ego':    torch.from_numpy(sample['hist_ego']),
                'hist_target': torch.from_numpy(sample['hist_target']),
                'hist_others': torch.from_numpy(sample['hist_others']),
                'mask_others': torch.from_numpy(sample['mask_others']),
                'context':     torch.from_numpy(sample['context']),
                'future_pos':  torch.from_numpy(sample['future_pos']),
            }
        # 7) print neighbor-count stats if we collected any
        if self._neighbor_counts:
            print("Neighbor count stats:",
                  "min=", np.min(self._neighbor_counts),
                  "median=", np.median(self._neighbor_counts),
                  "max=", np.max(self._neighbor_counts))
        else:
            print("Neighbor count stats: no neighbors were counted.")

    def _prepare_samples(self):
        samples = []
        T = self.timestamps
        F = len(self.features)
        total = len(T) - self.seq_len - self.pred_len + 1

        for i in tqdm(range(total), desc="Building samples", unit="win"):
            hist_times  = T[i : i + self.seq_len]
            target_time = hist_times[-1]
            future_time = T[i + self.seq_len - 1 + self.pred_len]
            if future_time not in self.social_data:
                continue

            ids_targ, feats_targ, idx_map_targ = self.social_data[target_time]
            context_feat = self.context[i + self.seq_len - 1]

            # loop over each neighbor ID
            for tid in ids_targ:
                # --- 1) target history ---
                hist_target = []
                ok = True
                for t in hist_times:
                    ids_t, feats_t, idx_map_t = self.social_data[t]
                    if tid not in idx_map_t:
                        ok = False
                        break
                    raw = feats_t[idx_map_t[tid]].copy()
                    raw = (raw - self.feat_mean.flatten()) / self.feat_std.flatten()
                    hist_target.append(raw)
                if not ok:
                    continue
                hist_target = np.stack(hist_target)

                # --- 2) top-K nearest others ---
                hist_others = []
                mask_others = []
                for t in hist_times:
                    ids_t, feats_t, _ = self.social_data[t]
                    # distances since x,y already ego-centric
                    dists_all = np.linalg.norm(feats_t[:, :2], axis=1)
                    idx_sorted = np.argsort(dists_all)
                    pick       = idx_sorted[:self.max_neighbors]
                    selected   = feats_t[pick]
                    self._neighbor_counts.append(len(pick))

                    selected = (selected - self.feat_mean) / self.feat_std

                    c = selected.shape[0]
                    if c < self.max_neighbors:
                        pad    = np.zeros((self.max_neighbors-c, F), dtype=selected.dtype)
                        padded = np.vstack([selected, pad])
                        mask_k = [1]*c + [0]*(self.max_neighbors-c)
                    else:
                        padded = selected
                        mask_k = [1]*self.max_neighbors

                    hist_others.append(padded)
                    mask_others.append(mask_k)

                hist_others = np.stack(hist_others)
                mask_others = np.array(mask_others, int)

                # --- 3) ego history ---
                hist_ego = []
                for t in hist_times:
                    raw_ego = self.ego_data[t].copy()
                    raw_ego = (raw_ego - self.feat_mean.flatten()) / self.feat_std.flatten()
                    hist_ego.append(raw_ego)
                hist_ego = np.stack(hist_ego)

                # --- 4) future position ---
                ids_fut, feats_fut, idx_map_fut = self.social_data[future_time]
                if tid not in idx_map_fut:
                    continue
                raw_fut    = feats_fut[idx_map_fut[tid]].copy()
                raw_fut    = (raw_fut - self.feat_mean.flatten()) / self.feat_std.flatten()
                future_pos = raw_fut[:2]

                samples.append({
                    'hist_ego':    hist_ego.astype(np.float32),
                    'hist_target': hist_target.astype(np.float32),
                    'hist_others': hist_others.astype(np.float32),
                    'mask_others': mask_others.astype(np.float32),
                    'context':     context_feat.astype(np.float32),
                    'future_pos':  future_pos.astype(np.float32),
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['hist_ego'],
            s['hist_target'],
            s['hist_others'],
            s['mask_others'],
            s['context'],
            s['future_pos'],
        )
