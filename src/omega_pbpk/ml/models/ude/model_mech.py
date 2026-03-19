"""Mechanistic 1-cpt model with IVIVE: encoder → ADME params → Cmax.

Key improvement over Phase 1 (abstract F/Vd/ka/ke):
  - Predicts MECHANISTIC parameters: CLint_hep, CLint_gut, fup, logP, ka, CLr
  - IVIVE converts CLint → hepatic clearance (well-stirred model)
  - Gut wall first-pass: CLint_gut → Fg (well-stirred gut)
  - Tissue partition: logP → Kp → Vd (Berezhkovskiy simplified)
  - F = Fg × Fh, CL = CLh + CLr, ke = CL/Vd → 1-cpt Cmax

Differentiable, closed-form, fast (~0.1ms/drug), mechanistically interpretable.
Auxiliary CLint/fup heads directly match TDC data (same physical quantities).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ===== Physiological constants (70 kg adult) =====
Q_LIVER = 76.5  # Total liver blood flow (L/h): hepatic artery + portal
Q_GUT = 57.0  # Gut blood flow (L/h)

# Organ volumes (L) for Vd calculation
V_PLASMA = 3.0  # Plasma volume
V_LIVER = 1.8
V_KIDNEY = 0.31
V_MUSCLE = 28.0
V_ADIPOSE = 14.5
V_GUT = 0.3
V_REST = 13.0
V_LUNG = 0.5

# Tissue neutral lipid fractions (Poulin & Theil 2002)
FN_LIVER = 0.0135
FN_KIDNEY = 0.0121
FN_MUSCLE = 0.0100
FN_ADIPOSE = 0.7021
FN_GUT = 0.0275
FN_REST = 0.0200
FN_LUNG = 0.0217

# Tissue water fractions
FW_LIVER = 0.726
FW_KIDNEY = 0.769
FW_MUSCLE = 0.756
FW_ADIPOSE = 0.150
FW_GUT = 0.693
FW_REST = 0.750
FW_LUNG = 0.834


def mechanistic_cmax(
    clint_hep: torch.Tensor,
    clint_gut: torch.Tensor,
    fup: torch.Tensor,
    log_partition: torch.Tensor,
    ka: torch.Tensor,
    clr: torch.Tensor,
    dose: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Cmax from ADME parameters via IVIVE + 1-compartment model.

    All inputs are (B,) tensors.

    Mechanistic chain:
      CLint_hep → Fh (well-stirred liver)
      CLint_gut → Fg (well-stirred gut wall)
      F = Fg × Fh (oral bioavailability, assuming Fa=1)
      CL = CLh + CLr (total clearance)
      logP → Kp[tissues] → Vd (volume of distribution)
      ke = CL / Vd (elimination rate)
      Cmax via 1-cpt analytical formula

    Returns:
        cmax: (B,) Cmax in mg/L
        params: dict with intermediate PK parameters
    """
    # === First-pass metabolism (well-stirred model) ===
    # Fg = Q_gut / (Q_gut + fup × CLint_gut)
    Fg = Q_GUT / (Q_GUT + fup * clint_gut)
    Fg = Fg.clamp(min=0.01, max=1.0)

    # Fh = Q_liver / (Q_liver + fup × CLint_hep)
    Fh = Q_LIVER / (Q_LIVER + fup * clint_hep)
    Fh = Fh.clamp(min=0.01, max=1.0)

    # Oral bioavailability (assuming Fa = 1 for IR oral)
    F = Fg * Fh

    # === Clearance ===
    # CLh = Q_liver × Eh = Q_liver × (1 - Fh)
    CLh = Q_LIVER * (1 - Fh)
    CL = CLh + clr  # total clearance (L/h)
    CL = CL.clamp(min=0.01)

    # === Volume of distribution (from tissue Kp) ===
    P = torch.exp(log_partition).clamp(min=0.1, max=1e4)

    # Vd = V_plasma + Σ(V_tissue × Kp_tissue)
    # where Kp = fw + fn × P (simplified Berezhkovskiy)
    Vd = (
        V_PLASMA
        + V_LIVER * (FW_LIVER + FN_LIVER * P)
        + V_KIDNEY * (FW_KIDNEY + FN_KIDNEY * P)
        + V_MUSCLE * (FW_MUSCLE + FN_MUSCLE * P)
        + V_ADIPOSE * (FW_ADIPOSE + FN_ADIPOSE * P)
        + V_GUT * (FW_GUT + FN_GUT * P)
        + V_REST * (FW_REST + FN_REST * P)
        + V_LUNG * (FW_LUNG + FN_LUNG * P)
    )
    Vd = Vd.clamp(min=1.0, max=10000.0)

    # === Elimination ===
    ke = CL / Vd  # h⁻¹
    ke = ke.clamp(min=0.001, max=10.0)

    # === 1-compartment Cmax ===
    diff = ka - ke
    near_equal = diff.abs() < 0.01
    safe_diff = torch.where(near_equal, torch.full_like(diff, 0.01), diff)

    tmax = torch.log(ka / ke.clamp(min=1e-6)) / safe_diff
    tmax = tmax.clamp(min=0.01, max=72.0)

    cmax_std = F * dose * ka / (Vd * safe_diff) * (torch.exp(-ke * tmax) - torch.exp(-ka * tmax))
    cmax_deg = F * dose / (Vd * math.e)
    cmax = torch.where(near_equal, cmax_deg, cmax_std)

    return cmax.clamp(min=1e-10), {
        "F": F,
        "Fg": Fg,
        "Fh": Fh,
        "CL": CL,
        "CLh": CLh,
        "Vd": Vd,
        "ke": ke,
        "tmax": tmax,
    }


class MechanisticPKModel(nn.Module):
    """Mechanistic multi-task PK model: encoder → ADME → IVIVE → 1-cpt → Cmax.

    Shared encoder with output heads:
    - ADME head (6): CLint_hep, CLint_gut, fup, log_partition, ka, CLr
    - CLint auxiliary (1): for TDC CLint data
    - fup auxiliary (1): for TDC fup data
    """

    def __init__(self, fp_dim: int = 2048, desc_dim: int = 9):
        super().__init__()

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

        # ADME head: 6 mechanistic parameters
        self.adme_head = nn.Linear(32, 6)

        # Auxiliary heads (shared encoder)
        self.clint_head = nn.Linear(32, 1)
        self.fup_head = nn.Linear(32, 1)

        self._init_biases()

    def _init_biases(self) -> None:
        """Physical defaults for zero-input prediction."""
        with torch.no_grad():
            # CLint_hep: exp(2.3) ≈ 10 L/h
            self.adme_head.bias[0] = math.log(10.0)
            # CLint_gut: exp(0) = 1.0 L/h
            self.adme_head.bias[1] = 0.0
            # fup: sigmoid(0) = 0.5
            self.adme_head.bias[2] = 0.0
            # log_partition: 2.0 → P≈7.4 (moderate logP)
            self.adme_head.bias[3] = 2.0
            # ka: exp(0) = 1.0/h
            self.adme_head.bias[4] = 0.0
            # CLr: exp(-1) ≈ 0.37 L/h
            self.adme_head.bias[5] = -1.0
            # Auxiliary
            self.clint_head.bias[0] = 2.0
            self.fup_head.bias[0] = 0.0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        fp = x[:, :2048]
        desc = x[:, 2048:]
        return self.shared(torch.cat([self.fp_encoder(fp), self.desc_encoder(desc)], dim=1))

    def _decode_adme(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        """Raw head output → physical ADME parameters."""
        return {
            "clint_hep": torch.exp(raw[:, 0]).clamp(min=0.01, max=500.0),
            "clint_gut": torch.exp(raw[:, 1]).clamp(min=0.001, max=100.0),
            "fup": torch.sigmoid(raw[:, 2]).clamp(min=0.001, max=0.999),
            "log_partition": raw[:, 3].clamp(min=-2.0, max=8.0),
            "ka": torch.exp(raw[:, 4]).clamp(min=0.01, max=20.0),
            "clr": torch.exp(raw[:, 5]).clamp(min=0.001, max=50.0),
        }

    def predict_cmax(
        self, x: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """SMILES features → ADME → IVIVE → Cmax."""
        h = self._encode(x)
        raw = self.adme_head(h)
        adme = self._decode_adme(raw)

        cmax, pk = mechanistic_cmax(
            adme["clint_hep"],
            adme["clint_gut"],
            adme["fup"],
            adme["log_partition"],
            adme["ka"],
            adme["clr"],
            dose,
        )
        return cmax, {**adme, **pk}

    def predict_clint(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode(x)
        return torch.exp(self.clint_head(h).squeeze(-1))

    def predict_fup(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode(x)
        return torch.sigmoid(self.fup_head(h).squeeze(-1))
