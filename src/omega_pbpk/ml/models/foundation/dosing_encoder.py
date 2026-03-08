"""Dosing regimen encoder for Level 3 foundation model.

Encodes dose amount, route, frequency, formulation, and infusion duration
into a 64-dim embedding for cross-attention fusion with molecular features.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

# Default dosing regimen (single oral dose, immediate release)
DOSING_DEFAULTS: dict[str, Any] = {
    "dose_mg": 100.0,
    "route": "oral",
    "frequency": "single",
    "duration_h": 0.0,
    "n_doses": 1,
    "formulation": "immediate_release",
}

# Continuous dosing features
DOSING_CONTINUOUS = ("dose_mg", "duration_h", "n_doses")

# Categorical dosing feature definitions
DOSING_CATEGORICAL: dict[str, tuple[str, ...]] = {
    "route": ("oral", "iv_bolus", "iv_infusion", "subcutaneous", "intramuscular"),
    "frequency": ("single", "BID", "TID", "QD", "Q12H", "Q8H", "Q6H", "continuous"),
    "formulation": (
        "solution",
        "immediate_release",
        "extended_release",
        "suspension",
    ),
}


class DosingEncoder(nn.Module):
    """Encode dosing regimen into a fixed-size embedding.

    Architecture:
        Continuous (dose_mg, duration_h, n_doses) -> log-transform -> Linear(3, 32) -> ReLU
        Categorical (route, frequency, formulation) -> Embeddings -> concat -> Linear(total, 32) -> ReLU
        Combined -> Linear(64, 64) -> ReLU -> Linear(64, 64) -> 64-dim

    Parameters
    ----------
    hidden_dim : int
        Per-branch hidden dimension (default: 32).
    output_dim : int
        Output embedding dimension (default: 64).
    cat_embedding_dim : int
        Embedding dimension per categorical variable (default: 8).
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 64,
        cat_embedding_dim: int = 8,
    ) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.cat_embedding_dim = cat_embedding_dim

        n_continuous = len(DOSING_CONTINUOUS)

        # Continuous branch (log-transformed inputs)
        self.cont_linear = nn.Linear(n_continuous, hidden_dim)
        self.cont_relu = nn.ReLU()

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for name, values in DOSING_CATEGORICAL.items():
            n_categories = len(values)
            self.cat_embeddings[name] = nn.Embedding(n_categories, cat_embedding_dim)
            total_cat_dim += cat_embedding_dim

        self.cat_linear = nn.Linear(total_cat_dim, hidden_dim)
        self.cat_relu = nn.ReLU()

        # Fusion: 2 * hidden_dim -> output_dim
        combined_dim = 2 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def _get_continuous_tensor(self, regimen: dict[str, Any], device: torch.device) -> torch.Tensor:
        """Extract and log-transform continuous dosing features.

        Returns
        -------
        Tensor of shape (1, 3)
        """
        values = []
        for feat in DOSING_CONTINUOUS:
            val = float(regimen.get(feat, DOSING_DEFAULTS[feat]))
            # log1p transform for numerical stability (handles 0 for duration)
            values.append(math.log1p(max(val, 0.0)))
        return torch.tensor([values], dtype=torch.float32, device=device)

    def _get_categorical_indices(
        self, regimen: dict[str, Any], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map categorical string values to embedding indices.

        Returns
        -------
        dict mapping feature name to LongTensor of shape (1,)
        """
        indices: dict[str, torch.Tensor] = {}
        for name, values in DOSING_CATEGORICAL.items():
            raw = regimen.get(name, DOSING_DEFAULTS[name])
            if raw in values:
                idx = values.index(raw)
            else:
                idx = values.index(DOSING_DEFAULTS[name])
            indices[name] = torch.tensor([idx], dtype=torch.long, device=device)
        return indices

    def forward(self, regimen: dict[str, Any]) -> torch.Tensor:
        """Encode dosing regimen.

        Parameters
        ----------
        regimen : dict
            Dosing regimen. Missing keys are filled with defaults.

        Returns
        -------
        Tensor of shape (batch_size, 64)
        """
        device = next(self.parameters()).device

        # Check if already tensorized
        if "continuous" in regimen and isinstance(regimen["continuous"], torch.Tensor):
            cont = regimen["continuous"].to(device)
            cat_indices = regimen["categorical"]
        else:
            cont = self._get_continuous_tensor(regimen, device)
            cat_indices = self._get_categorical_indices(regimen, device)

        # Continuous branch (already log-transformed)
        cont_out = self.cont_relu(self.cont_linear(cont))  # (batch, 32)

        # Categorical branch
        cat_parts = []
        for name in DOSING_CATEGORICAL:
            idx = cat_indices[name].to(device)
            emb = self.cat_embeddings[name](idx)  # (batch, cat_embedding_dim)
            cat_parts.append(emb)
        cat_concat = torch.cat(cat_parts, dim=-1)
        cat_out = self.cat_relu(self.cat_linear(cat_concat))  # (batch, 32)

        # Fuse
        combined = torch.cat([cont_out, cat_out], dim=-1)  # (batch, 64)
        return self.fusion(combined)  # (batch, 64)


__all__ = [
    "DosingEncoder",
    "DOSING_DEFAULTS",
    "DOSING_CONTINUOUS",
    "DOSING_CATEGORICAL",
]
