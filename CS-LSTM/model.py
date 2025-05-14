import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualSocialLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 context_dim: int,
                 intent_dim:  int = 0,
                 hidden_dim: int = 128,
                 max_neighbors: int = 10,
                 num_heads: int = 4,
                 pred_len: int = 1):
        """
        input_dim   = number of features per agent (e.g. 9)
        context_dim = dimension of your contextual_features vector
        hidden_dim  = LSTM hidden size
        max_neighbors = maximum neighbors per timestamp
        num_heads   = heads for MultiheadAttention
        pred_len    = number of steps to predict
        """
        super().__init__()
        self.pred_len      = pred_len
        self.hidden_dim    = hidden_dim
        self.max_neighbors = max_neighbors
        self.intent_dim   = intent_dim

        # 1) Encode ego history
        self.ego_lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 2) Encode target neighbor history
        self.target_lstm   = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # 3) Encode all neighbors
        self.neighbor_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # 4) Spatial attention over neighbors
        self.nbr_attn      = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 5) Embed context features
        self.context_fc    = nn.Linear(context_dim, hidden_dim)
        # 6b) intention embedding (if requested)
        if intent_dim > 0:
            self.intent_fc = nn.Linear(intent_dim, hidden_dim)
            fusion_dim = 5 * hidden_dim   # ego, targ, nbr, ctx, intent
        else:
            self.intent_fc = None
            fusion_dim = 4 * hidden_dim
        # 6) Fuse and predict first step
        #    cat of [ego_enc, targ_enc, nbr_enc, ctx_enc] = 4*hidden_dim
        self.output_fc     = nn.Linear(fusion_dim, 2)
        self.dropout       = nn.Dropout(0.1)

        # 7) Sequence‐to‐sequence decoder for pred_len > 1
        self.decoder_lstm  = nn.LSTM(2, hidden_dim, batch_first=True)
        ### CHANGED ### multi‐head outputs, one per intent class
        self.out_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2)
            for _ in range(3)
        ])

    def forward(self,
                hist_ego:    torch.Tensor,  # [B × T × D]
                hist_target: torch.Tensor,  # [B × T × D]
                hist_others: torch.Tensor,  # [B × T × N × D]
                mask_others: torch.Tensor,  # [B × T × N]
                context:     torch.Tensor,   # [B × context_dim]
                intent:      torch.Tensor=None   # [B × 3] one‐hot, or None
                ) -> torch.Tensor:             # [B × pred_len × 2]
        B, T, N, D = hist_others.size()

        # — 1) Ego encoding —
        out_e, _   = self.ego_lstm(hist_ego)           # [B × T × H]
        ego_enc    = out_e[:, -1, :]                   # [B × H]

        # — 2) Target neighbor encoding —
        out_t, _   = self.target_lstm(hist_target)     # [B × T × H]
        targ_enc   = out_t[:, -1, :]                   # [B × H]

        # — 3) Neighbor encoding & spatial attention —
        nbr_flat, _ = hist_others.view(B*N, T, D), None
        out_n, _    = self.neighbor_lstm(nbr_flat)     # [(B×N) × T × H]
        final_n     = out_n[:, -1, :].view(B, N, -1)   # [B × N × H]

        # mask out absent neighbors
        m_last      = mask_others[:, -1, :].unsqueeze(2)  # [B × N × 1]
        # apply attention: query=target_enc, key/value=neighbors
        attn_out, _ = self.nbr_attn(
            query=targ_enc.unsqueeze(1),  # [B × 1 × H]
            key=final_n,                  # [B × N × H]
            value=final_n
        )
        nbr_enc     = attn_out.squeeze(1)  # [B × H]

        # — 4) Context embedding —
        ctx_enc     = F.relu(self.context_fc(context))   # [B × H]
        # — 4b) Intention encoding (if present) —
        intent_enc  = None
        if self.intent_fc is not None:
            intent_enc = F.relu(self.intent_fc(intent))  # [B × H]        
        # — 5) Fuse & first‐step prediction —
        to_cat = [ego_enc, targ_enc, nbr_enc, ctx_enc]
        if intent_enc is not None:
            to_cat.append(intent_enc)

        cat = torch.cat(to_cat, dim=1)           # [B × (4H or 5H)]

        cat = self.dropout(cat)
        pred_init   = self.output_fc(cat)               # [B × 2]
        # if we only need one step, bypass the untrained decoder heads
        if self.pred_len == 1:
            # [B × 2] → [B × 1 × 2]
            return pred_init.unsqueeze(1)
        # — 6) Sequence‐to‐sequence decoding —
        #    feed first‐step pred back in for pred_len steps
        outputs = []
        dec_input = pred_init.unsqueeze(1)  # [B × 1 × 2]
        h, c      = None, None
        for _ in range(self.pred_len):
            if h is None:
                # first step: let the LSTM use its default zero‐state
                out_d, (h, c) = self.decoder_lstm(dec_input)
            else:
                # subsequent steps: feed back the state
                out_d, (h, c) = self.decoder_lstm(dec_input, (h, c))

            # multi‐head:
            #  → list of [B × 1 × 2], one per intent
            head_outs = [
                head(out_d[:, -1, :]).unsqueeze(1)
                for head in self.out_heads
            ]
            # stack into [B × 3 × 1 × 2]
            stacked = torch.stack(head_outs, dim=1)
            if intent is None:
                # average across modes → [B × 1 × 2]
                step_d = stacked.mean(dim=1)
            else:
                # intent: [B × 3]
                it = intent.view(B, 3, 1, 1)
                # pick the correct head per sample
                step_d = (stacked * it).sum(dim=1)            
            
            outputs.append(step_d)
            dec_input = step_d
        # concatenate along time
        pred_seq = torch.cat(outputs, dim=1)  # [B × pred_len × 2]
        return pred_seq
