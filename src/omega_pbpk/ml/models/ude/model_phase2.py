"""Phase 2 UDE model: MLP encoder → ADME parameters → differentiable PBPK ODE → Cmax.

Architecture:
  Morgan FP (2048) → Linear(64) → ReLU → Dropout(0.2)
  RDKit desc (9)   → Linear(16) → ReLU
  Concat(80) → Linear(32) → ReLU → Dropout(0.15)
  → ADME Head (7): CLint_hep, CLint_gut, fup, log_partition, ka, CLr, ps_base
  → Kp[7 tissues] via Berezhkovskiy(log_partition)
  → DifferentiablePBPK(13 states) via torchdiffeq
  → Cmax = max(C_venous(t))

Auxiliary heads (shared encoder):
  → CLint Head (1): for TDC CLint auxiliary loss
  → fup Head (1): for TDC fup auxiliary loss
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from omega_pbpk.ml.models.ude.pbpk_ode import DifferentiablePBPK, compute_kp


class UDEPhase2Model(nn.Module):
    """End-to-end SMILES → ADME → PBPK ODE → Cmax model."""

    def __init__(self, fp_dim: int = 2048, desc_dim: int = 9):
        super().__init__()

        # Shared encoder (same architecture as Phase 1)
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, 16),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        # ADME head: 7 outputs
        # [log_clint_hep, log_clint_gut, logit_fup, log_partition, log_ka, log_clr, log_ps]
        self.adme_head = nn.Linear(32, 7)

        # Auxiliary heads (same as Phase 1)
        self.clint_head = nn.Linear(32, 1)
        self.fup_head = nn.Linear(32, 1)

        # PBPK ODE solver
        self.pbpk = DifferentiablePBPK(t_max=24.0, n_points=80)

        self._init_biases()

    def _init_biases(self) -> None:
        """Initialize ADME head biases to physical defaults."""
        with torch.no_grad():
            # CLint_hep: exp(2.3) ≈ 10 L/h (moderate hepatic clearance)
            self.adme_head.bias[0] = math.log(10.0)
            # CLint_gut: exp(0) = 1.0 L/h (low gut metabolism)
            self.adme_head.bias[1] = 0.0
            # fup: sigmoid(0) = 0.5 (moderate binding)
            self.adme_head.bias[2] = 0.0
            # log_partition: 2.0 → P=7.4 (moderate lipophilicity)
            self.adme_head.bias[3] = 2.0
            # ka: exp(0) = 1.0/h (typical absorption)
            self.adme_head.bias[4] = 0.0
            # CLr: exp(-1) ≈ 0.37 L/h (low renal clearance)
            self.adme_head.bias[5] = -1.0
            # ps_base: exp(1.6) ≈ 5.0 L/h (typical PS)
            self.adme_head.bias[6] = math.log(5.0)
            # Auxiliary heads
            self.clint_head.bias[0] = 2.0
            self.fup_head.bias[0] = 0.0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: (B, 2057) → (B, 32)."""
        fp = x[:, :2048]
        desc = x[:, 2048:]
        h_fp = self.fp_encoder(fp)
        h_desc = self.desc_encoder(desc)
        h = torch.cat([h_fp, h_desc], dim=1)
        return self.shared(h)

    def _decode_adme(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode raw ADME head output to physical parameters.

        Args:
            raw: (B, 7) raw output from adme_head.

        Returns:
            Dict with all ADME parameters as (B,) tensors.
        """
        clint_hep = torch.exp(raw[:, 0]).clamp(min=0.01, max=500.0)  # L/h
        clint_gut = torch.exp(raw[:, 1]).clamp(min=0.001, max=100.0)  # L/h
        fup = torch.sigmoid(raw[:, 2]).clamp(min=0.001, max=0.999)  # (0, 1)
        log_partition = raw[:, 3].clamp(min=-2.0, max=8.0)  # log(P)
        ka = torch.exp(raw[:, 4]).clamp(min=0.01, max=20.0)  # h⁻¹
        clr = torch.exp(raw[:, 5]).clamp(min=0.001, max=50.0)  # L/h
        ps_base = torch.exp(raw[:, 6]).clamp(min=0.1, max=50.0)  # L/h

        kp = compute_kp(log_partition)

        return {
            "clint_hep": clint_hep,
            "clint_gut": clint_gut,
            "fup": fup,
            "log_partition": log_partition,
            "ka": ka,
            "clr": clr,
            "kp": kp,
            "ps_muscle": ps_base,
            "ps_adipose": ps_base * 0.4,  # adipose has lower PS
        }

    def predict_cmax(
        self, x: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict Cmax via full PBPK ODE.

        Args:
            x: (B, 2057) molecular features.
            dose: (B,) dose in mg.

        Returns:
            cmax: (B,) predicted Cmax in mg/L.
            params: dict with all ADME parameters + ODE outputs.
        """
        h = self._encode(x)
        raw = self.adme_head(h)
        adme = self._decode_adme(raw)

        cmax, ode_info = self.pbpk(
            dose,
            adme["clint_hep"],
            adme["clint_gut"],
            adme["fup"],
            adme["ka"],
            adme["clr"],
            adme["kp"],
            adme["ps_muscle"],
            adme["ps_adipose"],
        )

        params = {**adme, **ode_info}
        return cmax, params

    def predict_clint(self, x: torch.Tensor) -> torch.Tensor:
        """Predict CLint (µL/min/10^6 cells) via auxiliary head."""
        h = self._encode(x)
        return torch.exp(self.clint_head(h).squeeze(-1))

    def predict_fup(self, x: torch.Tensor) -> torch.Tensor:
        """Predict fraction unbound via auxiliary head."""
        h = self._encode(x)
        return torch.sigmoid(self.fup_head(h).squeeze(-1))
