"""Parameter prediction head that maps GNN embeddings to ODE parameters.

Produces named PK parameters with physically meaningful activations:
- Positive params (CLint, ka, Vd, ke): softplus
- Bounded params (fup, rbp, bioavailability): sigmoid variants
- Unbounded params (logP, logS): linear
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PKParameterHead(nn.Module):
    """Maps molecular embeddings to named PK parameters.

    Architecture:
        shared: embedding_dim -> 256 -> ReLU -> 128 -> ReLU
        per-group output heads with activation constraints

    Output dictionary keys match fields used by the Drug dataclass:
        - mw: molecular weight (g/mol), positive
        - logP: partition coefficient, unbounded
        - logS: log solubility, unbounded
        - fup: fraction unbound in plasma [0, 1]
        - rbp: blood-to-plasma ratio [0.5, 3.0]
        - peff: intestinal permeability (positive)
        - clint_3a4: CYP3A4 intrinsic clearance (positive, uL/min/pmol)
        - clint_2d6: CYP2D6 intrinsic clearance (positive, uL/min/pmol)
        - ka: absorption rate constant (positive, h^-1)
        - vd: volume of distribution (positive, L)
        - ke: elimination rate constant (positive, h^-1)
        - bioavailability: oral bioavailability [0, 1]

    Parameters
    ----------
    embedding_dim : int
        Input embedding dimension (default: 256).
    hidden_dim : int
        Shared hidden layer dimension (default: 128).
    """

    # Parameter group definitions: (name, count, activation_type)
    POSITIVE_PARAMS = ("mw", "peff", "clint_3a4", "clint_2d6", "ka", "vd", "ke")
    SIGMOID_PARAMS = ("fup", "bioavailability")  # [0, 1]
    RBP_PARAMS = ("rbp",)  # [0.5, 3.0]
    UNBOUNDED_PARAMS = ("logP", "logS")

    ALL_PARAMS = POSITIVE_PARAMS + SIGMOID_PARAMS + RBP_PARAMS + UNBOUNDED_PARAMS

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

        # Separate output heads per parameter group
        self.positive_head = nn.Linear(hidden_dim, len(self.POSITIVE_PARAMS))
        self.sigmoid_head = nn.Linear(hidden_dim, len(self.SIGMOID_PARAMS))
        self.rbp_head = nn.Linear(hidden_dim, len(self.RBP_PARAMS))
        self.unbounded_head = nn.Linear(hidden_dim, len(self.UNBOUNDED_PARAMS))

        self._softplus = nn.Softplus()

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map molecular embedding to named PK parameters.

        Parameters
        ----------
        embedding : Tensor of shape (batch_size, embedding_dim)

        Returns
        -------
        dict[str, Tensor]
            Each value has shape (batch_size,).
        """
        h = self.shared(embedding)

        result: dict[str, torch.Tensor] = {}

        # Positive parameters: softplus ensures > 0
        pos_raw = self.positive_head(h)
        pos_activated = self._softplus(pos_raw)
        for i, name in enumerate(self.POSITIVE_PARAMS):
            result[name] = pos_activated[:, i]

        # Sigmoid parameters: [0, 1]
        sig_raw = self.sigmoid_head(h)
        sig_activated = torch.sigmoid(sig_raw)
        for i, name in enumerate(self.SIGMOID_PARAMS):
            result[name] = sig_activated[:, i]

        # RBP: 0.5 + 2.5 * sigmoid(x) -> [0.5, 3.0]
        rbp_raw = self.rbp_head(h)
        rbp_activated = 0.5 + 2.5 * torch.sigmoid(rbp_raw)
        for i, name in enumerate(self.RBP_PARAMS):
            result[name] = rbp_activated[:, i]

        # Unbounded parameters: no activation
        unb_raw = self.unbounded_head(h)
        for i, name in enumerate(self.UNBOUNDED_PARAMS):
            result[name] = unb_raw[:, i]

        return result

    def parameter_names(self) -> list[str]:
        """Return list of all output parameter names."""
        return list(self.ALL_PARAMS)

    def to_drug_kwargs(self, params: dict[str, torch.Tensor], idx: int = 0) -> dict[str, Any]:
        """Convert predicted params at batch index `idx` to Drug-compatible kwargs.

        Parameters
        ----------
        params : dict from forward()
        idx : batch index

        Returns
        -------
        dict suitable for Drug(**kwargs)
        """
        return {
            "mw": params["mw"][idx].item(),
            "logP": params["logP"][idx].item(),
            "fup": params["fup"][idx].item(),
            "rbp": params["rbp"][idx].item(),
            "peff": params["peff"][idx].item(),
            "clint": {
                "CYP3A4": params["clint_3a4"][idx].item(),
                "CYP2D6": params["clint_2d6"][idx].item(),
            },
        }
