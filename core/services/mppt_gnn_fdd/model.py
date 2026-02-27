from __future__ import annotations

import torch
import torch.nn as nn


class GRUBlock(nn.Module):
    def __init__(self, fin: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru1 = nn.GRU(fin, hidden, batch_first=True)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,N,T,F]
        return: [B,N,H]
        """
        B, N, T, F = x.shape
        x2 = x.reshape(B * N, T, F)
        y1, _ = self.gru1(x2)
        y1 = self.drop(y1)
        y2, _ = self.gru2(y1)
        y = self.ln(y2 + y1)
        h = y[:, -1, :]                 # [B*N,H]
        return h.reshape(B, N, -1)


class EdgeNodeLayer(nn.Module):
    def __init__(self, d_node: int = 64, d_edge_in: int = 4, d_edge: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp_e = nn.Sequential(
            nn.Linear(d_edge_in + 2 * d_node, d_edge),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_edge, d_edge),
            nn.ReLU(),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(d_node + d_edge, d_node),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_node, d_node),
            nn.ReLU(),
        )

    def forward(self, v: torch.Tensor, e_attr: torch.Tensor) -> torch.Tensor:
        """
        v: [B,N,D]
        e_attr: [B,N,N,Fe]
        """
        B, N, D = v.shape
        vi = v[:, :, None, :].expand(B, N, N, D)
        vj = v[:, None, :, :].expand(B, N, N, D)
        inp = torch.cat([e_attr, vi, vj], dim=-1)      # [B,N,N,Fe+2D]
        e_upd = self.mlp_e(inp)                         # [B,N,N,De]

        # zera self-loop
        eye = torch.eye(N, device=v.device, dtype=torch.bool)[None, :, :, None]
        e_upd = e_upd.masked_fill(eye, 0.0)

        m = e_upd.sum(dim=2)                            # [B,N,De]
        v2 = self.mlp_v(torch.cat([v, m], dim=-1))       # [B,N,D]
        return v2


class MPPTGNNFDD(nn.Module):
    def __init__(self, fin_ts: int, fe: int, n_classes: int, d: int = 64):
        super().__init__()
        self.enc = GRUBlock(fin_ts, hidden=d)
        self.gnn1 = EdgeNodeLayer(d_node=d, d_edge_in=fe)
        self.gnn2 = EdgeNodeLayer(d_node=d, d_edge_in=fe)
        self.head = nn.Linear(d, n_classes)

    def forward(self, X_ts: torch.Tensor, E_attr: torch.Tensor) -> torch.Tensor:
        """
        X_ts: [B,N,T,F]
        E_attr: [B,N,N,Fe]
        logits: [B,N,C]
        """
        v = self.enc(X_ts)
        v = self.gnn1(v, E_attr)
        v = self.gnn2(v, E_attr)
        return self.head(v)