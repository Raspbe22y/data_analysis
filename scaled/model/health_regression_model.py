import torch
import torch.nn as nn

class TransformerEncoderWithMLP(nn.Module):
    def __init__(
        self,
        seq_len: int = 300,
        in_dim: int = 17,
        out_dim: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__()

        assert pooling in ["mean", "cls"]

        self.pooling = pooling

        # Input mapping d_model
        self.input_proj = nn.Linear(in_dim, d_model)

        # Choosable CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            seq_len += 1

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, S, D)
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)

        # MLP decoder head
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 300, 17)
        """
        B, S, _ = x.shape

        x = self.input_proj(x)  # (B, S, d_model)

        if self.pooling == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # Pooling
        if self.pooling == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        return self.mlp_head(x)
