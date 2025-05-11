import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextualSocialLSTM(nn.Module):
    def __init__(self,
                 input_dim: int,
                 context_dim: int,
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

        # 6) Fuse and predict first step
        #    cat of [ego_enc, targ_enc, nbr_enc, ctx_enc] = 4*hidden_dim
        self.output_fc     = nn.Linear(4 * hidden_dim, 2)
        self.dropout       = nn.Dropout(0.1)

        # 7) Sequence‐to‐sequence decoder for pred_len > 1
        self.decoder_lstm  = nn.LSTM(2, hidden_dim, batch_first=True)
        self.out_step      = nn.Linear(hidden_dim, 2)


    def forward(self,
                hist_ego:    torch.Tensor,  # [B × T × D]
                hist_target: torch.Tensor,  # [B × T × D]
                hist_others: torch.Tensor,  # [B × T × N × D]
                mask_others: torch.Tensor,  # [B × T × N]
                context:     torch.Tensor   # [B × context_dim]
               ) -> torch.Tensor:            # [B × pred_len × 2]
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

        # — 5) Fuse & first‐step prediction —
        cat         = torch.cat([ego_enc, targ_enc, nbr_enc, ctx_enc], dim=1)  # [B × 4H]
        cat         = self.dropout(cat)
        pred_init   = self.output_fc(cat)               # [B × 2]

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

            step_d    = self.out_step(out_d[:, -1, :]).unsqueeze(1)  # [B × 1 × 2]
            outputs.append(step_d)
            dec_input = step_d
        # concatenate along time
        pred_seq = torch.cat(outputs, dim=1)  # [B × pred_len × 2]
        return pred_seq
