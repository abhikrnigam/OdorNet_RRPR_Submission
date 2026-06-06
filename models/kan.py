import torch
import torch.nn as nn


class KANLayer(nn.Module):
    """
    Simple KAN layer: sum of learned 1D nonlinear functions φ_i(x_i)
    """
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # One small MLP per input dimension (spline-like)
        self.functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
            for _ in range(in_dim)
        ])

    def forward(self, x):
        # x: (B, in_dim)
        outs = []
        for i, fn in enumerate(self.functions):
            xi = x[:, i:i+1]   # (B, 1)
            outs.append(fn(xi))
        return torch.stack(outs, dim=0).sum(dim=0)  # (B, out_dim)
