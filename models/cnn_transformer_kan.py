import torch
import torch.nn as nn
from models.kan import KANLayer


class CNNTransformerKAN(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels=8,
        cnn_out_dim=64,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.4
    ):
        super().__init__()

        # -------- 1D CNN --------
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, cnn_out_dim, kernel_size=3, padding=1),
            nn.GELU()
        )

        self.cnn_proj = nn.Linear(cnn_out_dim, d_model)

        # -------- Transformer --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.norm = nn.LayerNorm(d_model)

        # -------- KAN Head --------
        self.kan = KANLayer(
            in_dim=d_model,
            out_dim=num_classes,
            hidden_dim=32
        )

    def forward(self, x):
        """
        x: (B, P, T, C)
        """
        B, P, T, C = x.shape

        # (B*P, C, T)
        x = x.view(B * P, T, C).transpose(1, 2)

        # CNN
        x = self.cnn(x)                  # (B*P, cnn_out_dim, T)
        x = x.mean(dim=-1)               # temporal pooling → (B*P, cnn_out_dim)

        # Project to transformer dim
        x = self.cnn_proj(x)             # (B*P, d_model)

        # Restore patches
        x = x.view(B, P, -1)              # (B, P, d_model)

        # Transformer
        x = self.transformer(x)           # (B, P, d_model)
        x = self.norm(x)

        # Global pooling over patches
        x = x.mean(dim=1)                 # (B, d_model)

        # KAN classification head
        logits = self.kan(x)              # (B, num_classes)
        return logits
