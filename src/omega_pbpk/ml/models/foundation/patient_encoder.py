"""Patient covariate encoder for Level 3 foundation model.

Maps patient demographics, genotype, and disease states to a 64-dim
embedding that modulates drug PK predictions via cross-attention fusion.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# Population defaults (healthy adult reference)
POPULATION_DEFAULTS: dict[str, Any] = {
    # Continuous
    "age": 40.0,
    "weight": 70.0,
    "height": 170.0,
    "bmi": 24.2,
    "egfr": 100.0,
    "albumin": 4.0,
    "hematocrit": 0.45,
    # Categorical
    "sex": "M",
    "race": "white",
    "cyp2d6_genotype": "EM",
    "cyp3a4_genotype": "NM",
    # Disease states
    "hepatic_impairment": "none",
    "renal_impairment": "none",
}

# Continuous feature names in canonical order
CONTINUOUS_FEATURES = (
    "age",
    "weight",
    "height",
    "bmi",
    "egfr",
    "albumin",
    "hematocrit",
)

# Categorical feature definitions: name -> list of valid values
CATEGORICAL_FEATURES: dict[str, tuple[str, ...]] = {
    "sex": ("M", "F"),
    "race": ("white", "black", "asian", "hispanic", "other"),
    "cyp2d6_genotype": ("PM", "IM", "EM", "UM"),
    "cyp3a4_genotype": ("PM", "NM"),
    "hepatic_impairment": ("none", "mild", "moderate", "severe"),
    "renal_impairment": ("none", "mild", "moderate", "severe", "ESRD"),
}


class PatientEncoder(nn.Module):
    """Encode patient covariates into a fixed-size embedding.

    Architecture:
        Continuous features -> LayerNorm -> Linear(7, 64) -> ReLU
        Categorical features -> Embedding layers -> concat -> Linear(total, 64) -> ReLU
        Combined -> Linear(128, 128) -> ReLU -> Linear(128, 64) -> 64-dim

    Missing covariates are filled with population defaults (healthy 70 kg, 40 y male).

    Parameters
    ----------
    continuous_dim : int
        Number of continuous features (default: 7).
    hidden_dim : int
        Intermediate hidden dimension (default: 128).
    output_dim : int
        Output embedding dimension (default: 64).
    cat_embedding_dim : int
        Embedding dimension per categorical variable (default: 8).
    """

    def __init__(
        self,
        continuous_dim: int = len(CONTINUOUS_FEATURES),
        hidden_dim: int = 128,
        output_dim: int = 64,
        cat_embedding_dim: int = 8,
    ) -> None:
        super().__init__()

        self.continuous_dim = continuous_dim
        self.output_dim = output_dim
        self.cat_embedding_dim = cat_embedding_dim

        # Continuous branch
        self.cont_norm = nn.LayerNorm(continuous_dim)
        self.cont_linear = nn.Linear(continuous_dim, 64)
        self.cont_relu = nn.ReLU()

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for name, values in CATEGORICAL_FEATURES.items():
            n_categories = len(values)
            self.cat_embeddings[name] = nn.Embedding(n_categories, cat_embedding_dim)
            total_cat_dim += cat_embedding_dim

        self.cat_linear = nn.Linear(total_cat_dim, 64)
        self.cat_relu = nn.ReLU()

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _get_continuous_tensor(
        self, covariates: dict[str, Any], device: torch.device
    ) -> torch.Tensor:
        """Extract continuous features from covariates dict.

        Returns
        -------
        Tensor of shape (1, n_continuous)
        """
        values = []
        for feat in CONTINUOUS_FEATURES:
            val = covariates.get(feat, POPULATION_DEFAULTS[feat])
            values.append(float(val))
        return torch.tensor([values], dtype=torch.float32, device=device)

    def _get_categorical_indices(
        self, covariates: dict[str, Any], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map categorical string values to embedding indices.

        Returns
        -------
        dict mapping feature name to LongTensor of shape (1,)
        """
        indices: dict[str, torch.Tensor] = {}
        for name, values in CATEGORICAL_FEATURES.items():
            raw = covariates.get(name, POPULATION_DEFAULTS[name])
            if raw in values:
                idx = values.index(raw)
            else:
                # Fall back to default
                idx = values.index(POPULATION_DEFAULTS[name])
            indices[name] = torch.tensor([idx], dtype=torch.long, device=device)
        return indices

    def forward(self, covariates: dict[str, Any]) -> torch.Tensor:
        """Encode patient covariates.

        Parameters
        ----------
        covariates : dict
            Patient covariates. Missing keys are filled with population defaults.
            Can include batch of values as tensors or single values.

        Returns
        -------
        Tensor of shape (batch_size, 64)
        """
        device = next(self.parameters()).device

        # Check if input is already tensorized (batch mode)
        if "continuous" in covariates and isinstance(
            covariates["continuous"], torch.Tensor
        ):
            cont = covariates["continuous"].to(device)
            cat_indices = covariates["categorical"]
        else:
            cont = self._get_continuous_tensor(covariates, device)
            cat_indices = self._get_categorical_indices(covariates, device)

        # Continuous branch
        cont_normed = self.cont_norm(cont)
        cont_out = self.cont_relu(self.cont_linear(cont_normed))  # (batch, 64)

        # Categorical branch
        cat_parts = []
        for name in CATEGORICAL_FEATURES:
            idx = cat_indices[name].to(device)
            emb = self.cat_embeddings[name](idx)  # (batch, cat_embedding_dim)
            cat_parts.append(emb)
        cat_concat = torch.cat(cat_parts, dim=-1)  # (batch, total_cat_dim)
        cat_out = self.cat_relu(self.cat_linear(cat_concat))  # (batch, 64)

        # Fuse
        combined = torch.cat([cont_out, cat_out], dim=-1)  # (batch, 128)
        return self.fusion(combined)  # (batch, 64)


__all__ = [
    "PatientEncoder",
    "POPULATION_DEFAULTS",
    "CONTINUOUS_FEATURES",
    "CATEGORICAL_FEATURES",
]
