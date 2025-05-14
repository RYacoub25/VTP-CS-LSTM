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
                 use_context: bool = True,
                 use_intention: bool = False,
                 features=None):
        """
        - use_delta_yaw:  if True, append Δyaw into each agent’s feature vector
        - use_context:    if False, zero‐out your context vector
        - use_intention:  if True, compute a 3‐way one‐hot [straight,left,right] per sample
        """
        self.seq_len       = seq_len
        self.pred_len      = pred_len
        self.max_neighbors = max_neighbors
        self.use_delta_yaw = use_delta_yaw
        self.use_context   = use_context
        self.use_intention = use_intention

        # 1) build feature list
        if features is None:
            features = ['x','y','z','vx','vy','vz','ax','ay','az']
            if use_delta_yaw:
                features.append('delta_yaw')
        self.features = features

        # 2) load ego, add delta_yaw if needed
        self.ego_df = pd.read_csv(ego_path).sort_values('timestamp_ns')
        if use_delta_yaw:
            self.ego_df['delta_yaw'] = self.ego_df['yaw'].diff().fillna(0.0)

        # 3) load social, add delta_yaw only if missing+requested
        self.social_df = pd.read_csv(social_path).sort_values(['track_uuid','timestamp_ns'])
        if use_delta_yaw and 'delta_yaw' not in self.social_df:
            self.social_df['delta_yaw'] = (
                self.social_df
                    .groupby('track_uuid')['yaw']
                    .diff()
                    .fillna(0.0)
            )

        # 4) load context array
        self.context = np.load(contextual_path)

        # 5) group by timestamp
        self.ego_group    = self.ego_df.groupby('timestamp_ns')
        self.social_group = self.social_df.groupby('timestamp_ns')

        # 6) intersect timestamps
        ego_times    = set(self.ego_group.groups)
        social_times = set(self.social_group.groups)
        self.timestamps = sorted(ego_times & social_times)

        # 7) cache per‐timestamp social arrays
        self.social_data = {}
        for ts, df_ts in self.social_group:
            if ts not in social_times: continue
            ids     = df_ts['track_uuid'].values
            feats   = df_ts[self.features].values
            idx_map = {uid:i for i,uid in enumerate(ids)}
            self.social_data[ts] = (ids, feats, idx_map)

        # 8) cache per‐timestamp ego arrays
        self.ego_data = {
            ts: df_ts[self.features].values[0]
            for ts, df_ts in self.ego_group
            if ts in ego_times
        }

        # 9) cache raw yaw for intention logic
        if use_intention:
            self.social_yaw = {
                ts: df_ts['yaw'].values
                for ts, df_ts in self.social_group
                if ts in social_times
            }

        # 10) normalization stats
        all_feats = np.vstack([feats for _, feats, _ in self.social_data.values()])
        self.feat_mean = all_feats.mean(axis=0, keepdims=True)
        self.feat_std  = all_feats.std(axis=0, keepdims=True) + 1e-6

        # 11) build numpy samples (with progress bar)
        self._neighbor_counts = []
        self.samples = self._prepare_samples()

        # 12) convert all to torch.Tensor (no progress bar here)
        for i, s in enumerate(self.samples):
            self.samples[i] = {
                'hist_ego':    torch.from_numpy(s['hist_ego']),
                'hist_target': torch.from_numpy(s['hist_target']),
                'hist_others': torch.from_numpy(s['hist_others']),
                'mask_others': torch.from_numpy(s['mask_others']),
                'context':     torch.from_numpy(s['context']),
                'intent':      torch.from_numpy(s['intent']),
                'future_pos':  torch.from_numpy(s['future_pos']),
            }

        # 13) diagnostic
        if self._neighbor_counts:
            print("Neighbor count stats:",
                  "min=", np.min(self._neighbor_counts),
                  "median=", np.median(self._neighbor_counts),
                  "max=", np.max(self._neighbor_counts))
        else:
            print("Neighbor count stats: no neighbors counted.")

    def _prepare_samples(self):
        samples = []
        T = self.timestamps
        F = len(self.features)
        total = len(T) - self.seq_len - self.pred_len + 1

        zero_ctx = np.zeros_like(self.context[0]) if not self.use_context else None

        for i in tqdm(range(total), desc="Building samples", unit="win"):
            hist_times  = T[i : i + self.seq_len]
            target_time = hist_times[-1]
            future_time = T[i + self.seq_len - 1 + self.pred_len]
            if future_time not in self.social_data:
                continue

            ids_targ, feats_targ, idx_map_targ = self.social_data[target_time]
            ctx_feat = zero_ctx if zero_ctx is not None else self.context[i + self.seq_len - 1].copy()

            for tid in ids_targ:
                # 1) target history
                hist_target = []
                ok = True
                for t in hist_times:
                    ids_t, feats_t, idx_map_t = self.social_data[t]
                    if tid not in idx_map_t:
                        ok = False; break
                    raw = feats_t[idx_map_t[tid]].copy()
                    raw = (raw - self.feat_mean.flatten()) / self.feat_std.flatten()
                    hist_target.append(raw)
                if not ok: continue
                hist_target = np.stack(hist_target)

                # 2) neighbors
                hist_others = []; mask_others = []
                for t in hist_times:
                    ids_t, feats_t, _ = self.social_data[t]
                    dists = np.linalg.norm(feats_t[:, :2], axis=1)
                    pick  = np.argsort(dists)[:self.max_neighbors]
                    self._neighbor_counts.append(len(pick))
                    sel = feats_t[pick].copy()
                    sel = (sel - self.feat_mean) / self.feat_std
                    c = sel.shape[0]
                    if c < self.max_neighbors:
                        pad     = np.zeros((self.max_neighbors-c, F), dtype=sel.dtype)
                        sel     = np.vstack([sel, pad])
                        mask_k  = [1]*c + [0]*(self.max_neighbors-c)
                    else:
                        mask_k  = [1]*self.max_neighbors
                    hist_others.append(sel)
                    mask_others.append(mask_k)
                hist_others = np.stack(hist_others)
                mask_others = np.array(mask_others, dtype=np.float32)

                # 3) ego history
                hist_ego = []
                for t in hist_times:
                    raw_e = self.ego_data[t].copy()
                    raw_e = (raw_e - self.feat_mean.flatten()) / self.feat_std.flatten()
                    hist_ego.append(raw_e)
                hist_ego = np.stack(hist_ego)

                # 4) future Δpos
                ids_fut, feats_fut, idx_map_fut = self.social_data[future_time]
                if tid not in idx_map_fut: continue
                raw_fut = feats_fut[idx_map_fut[tid]].copy()
                raw_fut = (raw_fut - self.feat_mean.flatten()) / self.feat_std.flatten()
                future_pos = raw_fut[:2]

                # --- 5) intention one-hot ---
                if self.use_intention:
                    yaw_hist = []
                    for t in hist_times:
                        ids_t, feats_t, idx_map_t = self.social_data[t]
                        # now use the per-timestamp idx map!
                        yaw_hist.append(self.social_yaw[t][ idx_map_t[tid] ])
                    dy  = yaw_hist[-1] - yaw_hist[0]
                    dy  = ((dy + np.pi) % (2*np.pi)) - np.pi
                    thr = 0.2
                    cls = 0 if abs(dy)<=thr else (1 if dy>thr else 2)
                    intent = np.eye(3, dtype=np.float32)[cls]
                else:
                    intent = np.zeros(3, dtype=np.float32)


                samples.append({
                    'hist_ego':    hist_ego.astype(np.float32),
                    'hist_target': hist_target.astype(np.float32),
                    'hist_others': hist_others.astype(np.float32),
                    'mask_others': mask_others,
                    'context':     ctx_feat.astype(np.float32),
                    'intent':      intent,
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
            s['intent'],
            s['future_pos'],
        )
