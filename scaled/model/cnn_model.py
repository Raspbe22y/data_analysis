import torch
import torch.nn as nn


class CNNHealthModel(nn.Module):
    def __init__(
        self,
        seq_len: int = 300,
        in_dim: int = 17,
        out_dim: int = 6,
        num_filters: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.in_dim = in_dim

        # CNN encoder
        # Input shape: (B, seq_len, in_dim)
        # Conv1d expects: (B, in_dim, seq_len) so we transpose in forward()

        layers = []
        in_ch = in_dim
        out_ch = num_filters

        for i in range(num_layers):
            layers += [
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,  # same padding, keeps seq_len
                ),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)  # double filters each layer up to 256

        self.cnn = nn.Sequential(*layers)

        # Global average pooling over time dimension 
        # with output (B, final_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP head
        final_channels = in_ch
        self.mlp_head = nn.Sequential(
            nn.Linear(final_channels, final_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, in_dim)
        returns: (B, out_dim)
        """
        # Conv1d expects (B, channels, length)
        x = x.transpose(1, 2)          # (B, in_dim, seq_len)
        x = self.cnn(x)                 # (B, final_channels, seq_len)
        x = self.pool(x).squeeze(-1)    # (B, final_channels)
        return self.mlp_head(x)         # (B, out_dim)
