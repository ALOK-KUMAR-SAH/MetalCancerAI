from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetalCancerModel(nn.Module):
    """
    Multi-head regression model.

    Inputs:
      x: (batch, input_dim) float tensor

    Outputs:
      y_hat: (batch, 2) where:
        y_hat[:, 0] = cancer_risk_score in [0, 100]
        y_hat[:, 1] = months_to_onset in [0, +inf)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.20,
        output_dim: int = 2,  # kept for compatibility; must be 2 for this design
    ) -> None:
        super().__init__()
        if output_dim != 2:
            raise ValueError("MetalCancerModel is multi-head with output_dim fixed to 2.")

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*layers)

        # Two heads
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.onset_head = nn.Linear(hidden_dim, 1)

        # Small init tweak (optional, but often stabilizes early training)
        nn.init.xavier_uniform_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)
        nn.init.xavier_uniform_(self.onset_head.weight)
        nn.init.zeros_(self.onset_head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)

        # Risk: bound to [0, 100]
        risk_0_1 = torch.sigmoid(self.risk_head(h))
        risk_0_100 = 100.0 * risk_0_1

        # Months to onset: enforce non-negative
        months = F.softplus(self.onset_head(h))

        return torch.cat([risk_0_100, months], dim=1)