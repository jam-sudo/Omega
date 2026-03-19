"""Multi-task PK model: MLP encoder + 1-compartment analytical formula.

Architecture:
  Morgan FP (2048) -> Linear(64) -> ReLU -> Dropout(0.4)
  RDKit desc (9)   -> Linear(16) -> ReLU
  Concat(80) -> Linear(32) -> ReLU -> Dropout(0.3)
  -> PK Head (4): F, Vd, ka, ke via constrained activations
  -> CLint Head (1): exp activation
  -> fup Head (1): sigmoid activation

Total: ~134K trainable parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def safe_cmax_1cpt(
    F: torch.Tensor,
    dose: torch.Tensor,
    Vd: torch.Tensor,
    ka: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable 1-compartment oral Cmax.

    All inputs are (B,) tensors. Returns (B,) Cmax in mg/L.

    Handles ka ~ ke singularity via smooth switching.
    """
    # Clamp to physical ranges (preserves gradients via straight-through)
    Vd = Vd.clamp(min=0.5, max=5000.0)
    ka = ka.clamp(min=0.01, max=20.0)
    ke = ke.clamp(min=0.001, max=10.0)

    diff = ka - ke
    near_equal = diff.abs() < 0.01

    # Standard case: ka != ke
    safe_diff = torch.where(near_equal, torch.full_like(diff, 0.01), diff)
    tmax = torch.log(ka / ke) / safe_diff
    tmax = tmax.clamp(min=0.01, max=72.0)

    cmax_std = F * dose * ka / (Vd * safe_diff) * (torch.exp(-ke * tmax) - torch.exp(-ka * tmax))

    # Degenerate case: ka ~ ke -> Cmax = F * dose / (Vd * e)
    cmax_deg = F * dose / (Vd * math.e)

    cmax = torch.where(near_equal, cmax_deg, cmax_std)
    return cmax.clamp(min=1e-10)


class MultiTaskPKModel(nn.Module):
    """Multi-task PK prediction model.

    Shared encoder with three output heads:
    - PK head: F, Vd, ka, ke -> 1-cpt Cmax formula (primary)
    - CLint head: intrinsic clearance (auxiliary)
    - fup head: fraction unbound (auxiliary)
    """

    def __init__(self, fp_dim: int = 2048, desc_dim: int = 9):
        super().__init__()

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, 16),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # PK head: 4 outputs -> (F_logit, log_Vd, log_ka, log_ke)
        self.pk_head = nn.Linear(32, 4)

        # Auxiliary heads
        self.clint_head = nn.Linear(32, 1)
        self.fup_head = nn.Linear(32, 1)

        self._init_biases()

    def _init_biases(self) -> None:
        """Initialize PK head biases to physically reasonable defaults.

        Zero-input -> F=0.5, Vd=70L, ka=1.0/h, ke=0.1/h (t_half~7h).
        """
        with torch.no_grad():
            self.pk_head.bias[0] = 0.0  # sigmoid(0) = 0.5 -> F = 0.5
            self.pk_head.bias[1] = math.log(70.0)  # exp(4.25) ~ 70L
            self.pk_head.bias[2] = 0.0  # exp(0) = 1.0/h
            self.pk_head.bias[3] = math.log(0.1)  # exp(-2.3) ~ 0.1/h
            # CLint: exp(2) ~ 7.4 uL/min/10^6 (moderate)
            self.clint_head.bias[0] = 2.0
            # fup: sigmoid(0) = 0.5
            self.fup_head.bias[0] = 0.0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: (B, 2057) features -> (B, 32) latent."""
        fp = x[:, :2048]
        desc = x[:, 2048:]
        h_fp = self.fp_encoder(fp)
        h_desc = self.desc_encoder(desc)
        h = torch.cat([h_fp, h_desc], dim=1)
        return self.shared(h)

    def predict_cmax(
        self, x: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict Cmax via 1-compartment model.

        Args:
            x: (B, 2057) molecular features.
            dose: (B,) dose in mg.

        Returns:
            cmax: (B,) predicted Cmax in mg/L.
            params: dict with F, Vd, ka, ke tensors.
        """
        h = self._encode(x)
        raw = self.pk_head(h)  # (B, 4)

        F = torch.sigmoid(raw[:, 0])  # (0, 1)
        Vd = torch.exp(raw[:, 1])  # (0, inf) in L
        ka = torch.exp(raw[:, 2])  # (0, inf) in h^-1
        ke = torch.exp(raw[:, 3])  # (0, inf) in h^-1

        cmax = safe_cmax_1cpt(F, dose, Vd, ka, ke)

        return cmax, {"F": F, "Vd": Vd, "ka": ka, "ke": ke}

    def predict_clint(self, x: torch.Tensor) -> torch.Tensor:
        """Predict CLint (uL/min/10^6 cells), always positive."""
        h = self._encode(x)
        return torch.exp(self.clint_head(h).squeeze(-1))

    def predict_fup(self, x: torch.Tensor) -> torch.Tensor:
        """Predict fraction unbound in plasma, (0, 1)."""
        h = self._encode(x)
        return torch.sigmoid(self.fup_head(h).squeeze(-1))
