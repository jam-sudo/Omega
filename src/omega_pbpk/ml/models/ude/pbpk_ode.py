"""Differentiable 13-state PBPK ODE for end-to-end training.

Simplified whole-body model capturing essential PK mechanisms:
- ACAT absorption (single GI compartment)
- Gut wall first-pass (CYP3A4, determines Fg)
- Hepatic first-pass (well-stirred, determines Fh)
- Tissue distribution (perfusion + permeability-limited)
- Renal elimination

State vector (13):
  0: venous blood     1: arterial blood    2: lung
  3: liver            4: gut wall          5: kidney
  6: muscle vascular  7: muscle extravasc  8: adipose vascular
  9: adipose extravasc 10: rest (lumped)   11: GI lumen
  12: elimination sink

Physiological parameters from Poulin & Theil 2002, Brown 1997.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ===== Physiological constants (70 kg adult) =====

# Blood flows (L/h)
Q_CO = 390.0  # cardiac output
Q_LIVER_HA = 19.5  # hepatic artery (5% CO)
Q_GUT = 57.0  # gut (splanchnic → portal)
Q_LIVER_TOTAL = Q_LIVER_HA + Q_GUT  # 76.5
Q_KIDNEY = 72.0  # 18.5% CO
Q_MUSCLE = 42.9  # 11% CO
Q_ADIPOSE = 19.5  # 5% CO
Q_REST = Q_CO - Q_LIVER_HA - Q_GUT - Q_KIDNEY - Q_MUSCLE - Q_ADIPOSE  # 179.1

# Organ volumes (L)
V_VEN = 3.6
V_ART = 1.2
V_LUNG = 0.5
V_LIVER = 1.8
V_GUT = 0.3
V_KIDNEY = 0.31
V_MUSCLE = 28.0
V_ADIPOSE = 14.5
V_REST = 13.0

# Permeability-limited organ fractions
VASC_FRAC_MUSCLE = 0.04
VASC_FRAC_ADIPOSE = 0.031

# Tissue neutral lipid fractions (for Kp computation, Poulin & Theil 2002)
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

# State indices
IDX_VEN = 0
IDX_ART = 1
IDX_LUNG = 2
IDX_LIVER = 3
IDX_GUT = 4
IDX_KIDNEY = 5
IDX_MUSC_V = 6
IDX_MUSC_E = 7
IDX_ADIP_V = 8
IDX_ADIP_E = 9
IDX_REST = 10
IDX_GI = 11
IDX_ELIM = 12

N_STATES = 13


def compute_kp(log_partition: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute tissue:blood partition coefficients from lipophilicity.

    Uses simplified Berezhkovskiy: Kp = fw + fn * P
    where P = exp(log_partition) is the effective partition coefficient.

    Args:
        log_partition: (B,) log of effective partition coefficient.

    Returns:
        Dict mapping tissue name to (B,) Kp tensor.
    """
    P = torch.exp(log_partition).clamp(max=1e4)
    return {
        "liver": FW_LIVER + FN_LIVER * P,
        "kidney": FW_KIDNEY + FN_KIDNEY * P,
        "muscle": FW_MUSCLE + FN_MUSCLE * P,
        "adipose": FW_ADIPOSE + FN_ADIPOSE * P,
        "gut": FW_GUT + FN_GUT * P,
        "rest": FW_REST + FN_REST * P,
        "lung": FW_LUNG + FN_LUNG * P,
    }


class PBPKFunc(nn.Module):
    """ODE right-hand side for batched PBPK simulation.

    Drug-specific parameters are set via set_params() before each solve.
    The forward() method computes dy/dt for a batch of drugs.
    """

    def __init__(self):
        super().__init__()
        # Drug params set per-batch
        self.clint_hep: torch.Tensor = torch.zeros(1)
        self.clint_gut: torch.Tensor = torch.zeros(1)
        self.fup: torch.Tensor = torch.ones(1)
        self.ka: torch.Tensor = torch.ones(1)
        self.clr: torch.Tensor = torch.zeros(1)
        self.ps_muscle: torch.Tensor = torch.full((1,), 5.0)
        self.ps_adipose: torch.Tensor = torch.full((1,), 2.0)
        # Kp values (per tissue)
        self.kp_liver: torch.Tensor = torch.ones(1)
        self.kp_kidney: torch.Tensor = torch.ones(1)
        self.kp_muscle: torch.Tensor = torch.ones(1)
        self.kp_adipose: torch.Tensor = torch.ones(1)
        self.kp_gut: torch.Tensor = torch.ones(1)
        self.kp_rest: torch.Tensor = torch.ones(1)
        self.kp_lung: torch.Tensor = torch.ones(1)

    def set_params(
        self,
        clint_hep: torch.Tensor,
        clint_gut: torch.Tensor,
        fup: torch.Tensor,
        ka: torch.Tensor,
        clr: torch.Tensor,
        kp: dict[str, torch.Tensor],
        ps_muscle: torch.Tensor,
        ps_adipose: torch.Tensor,
    ) -> None:
        """Set drug-specific parameters for the current batch.

        All tensors are (B,) where B is the batch size.
        """
        self.clint_hep = clint_hep
        self.clint_gut = clint_gut
        self.fup = fup
        self.ka = ka
        self.clr = clr
        self.ps_muscle = ps_muscle
        self.ps_adipose = ps_adipose
        self.kp_liver = kp["liver"]
        self.kp_kidney = kp["kidney"]
        self.kp_muscle = kp["muscle"]
        self.kp_adipose = kp["adipose"]
        self.kp_gut = kp["gut"]
        self.kp_rest = kp["rest"]
        self.kp_lung = kp["lung"]

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute dy/dt for all 13 states.

        Args:
            t: scalar time (not used directly — autonomous ODE)
            y: (B, 13) state vector

        Returns:
            dydt: (B, 13) derivatives
        """
        dydt = torch.zeros_like(y)

        # === Extract amounts ===
        A_ven = y[:, IDX_VEN]
        A_art = y[:, IDX_ART]
        A_lung = y[:, IDX_LUNG]
        A_liver = y[:, IDX_LIVER]
        A_gut = y[:, IDX_GUT]
        A_kidney = y[:, IDX_KIDNEY]
        A_musc_v = y[:, IDX_MUSC_V]
        A_musc_e = y[:, IDX_MUSC_E]
        A_adip_v = y[:, IDX_ADIP_V]
        A_adip_e = y[:, IDX_ADIP_E]
        A_rest = y[:, IDX_REST]
        A_gi = y[:, IDX_GI]

        # === Blood concentrations (total, leaving each organ) ===
        C_ven = A_ven / V_VEN
        C_art = A_art / V_ART
        C_lung = A_lung / (V_LUNG * self.kp_lung)
        C_liver = A_liver / (V_LIVER * self.kp_liver)
        C_gut = A_gut / (V_GUT * self.kp_gut)
        C_kidney = A_kidney / (V_KIDNEY * self.kp_kidney)
        C_rest = A_rest / (V_REST * self.kp_rest)

        # Permeability-limited: vascular and extravascular
        v_musc_v = V_MUSCLE * VASC_FRAC_MUSCLE
        v_musc_e = V_MUSCLE * (1 - VASC_FRAC_MUSCLE)
        v_adip_v = V_ADIPOSE * VASC_FRAC_ADIPOSE
        v_adip_e = V_ADIPOSE * (1 - VASC_FRAC_ADIPOSE)

        C_musc_v = A_musc_v / v_musc_v
        C_musc_e = A_musc_e / v_musc_e
        C_adip_v = A_adip_v / v_adip_v
        C_adip_e = A_adip_e / v_adip_e

        # Unbound concentrations for PS transfer
        Cu_musc_v = self.fup * C_musc_v
        Cu_musc_e = self.fup * C_musc_e / self.kp_muscle.clamp(min=0.01)
        Cu_adip_v = self.fup * C_adip_v
        Cu_adip_e = self.fup * C_adip_e / self.kp_adipose.clamp(min=0.01)

        # === Absorption ===
        absorption = self.ka * A_gi.clamp(min=0)

        # === Metabolism ===
        liver_met = self.clint_hep * self.fup * C_liver
        gut_met = self.clint_gut * self.fup * C_gut

        # === Renal ===
        renal_elim = self.clr * C_kidney

        # === Permeability transfer ===
        ps_musc = self.ps_muscle * (Cu_musc_v - Cu_musc_e)
        ps_adip = self.ps_adipose * (Cu_adip_v - Cu_adip_e)

        # === ODEs ===

        # Venous blood: collects from all organs → to lung
        venous_return = (
            Q_LIVER_TOTAL * C_liver
            + Q_KIDNEY * C_kidney
            + Q_MUSCLE * C_musc_v
            + Q_ADIPOSE * C_adip_v
            + Q_REST * C_rest
        )
        dydt[:, IDX_VEN] = venous_return - Q_CO * C_ven

        # Arterial blood: from lung → distributes to all organs
        dydt[:, IDX_ART] = Q_CO * C_lung - Q_CO * C_art

        # Lung: venous → lung → arterial
        dydt[:, IDX_LUNG] = Q_CO * C_ven - Q_CO * C_lung

        # Liver: hepatic artery + portal (gut output) - outflow - metabolism
        dydt[:, IDX_LIVER] = (
            Q_LIVER_HA * C_art + Q_GUT * C_gut - Q_LIVER_TOTAL * C_liver - liver_met
        )

        # Gut wall: arterial + absorption - portal outflow - gut metabolism
        dydt[:, IDX_GUT] = Q_GUT * C_art + absorption - Q_GUT * C_gut - gut_met

        # Kidney: arterial - outflow - renal elimination
        dydt[:, IDX_KIDNEY] = Q_KIDNEY * (C_art - C_kidney) - renal_elim

        # Muscle (permeability-limited)
        dydt[:, IDX_MUSC_V] = Q_MUSCLE * (C_art - C_musc_v) - ps_musc
        dydt[:, IDX_MUSC_E] = ps_musc

        # Adipose (permeability-limited)
        dydt[:, IDX_ADIP_V] = Q_ADIPOSE * (C_art - C_adip_v) - ps_adip
        dydt[:, IDX_ADIP_E] = ps_adip

        # Rest (perfusion-limited, lumped)
        dydt[:, IDX_REST] = Q_REST * (C_art - C_rest)

        # GI lumen: drug leaves via absorption
        dydt[:, IDX_GI] = -absorption

        # Elimination sink (mass balance tracking)
        dydt[:, IDX_ELIM] = liver_met + gut_met + renal_elim

        return dydt


class DifferentiablePBPK(nn.Module):
    """Differentiable PBPK simulation via torchdiffeq.

    Solves the 13-state ODE and extracts Cmax from venous blood.

    Usage:
        pbpk = DifferentiablePBPK()
        cmax = pbpk(dose, clint_hep, clint_gut, fup, ka, clr, kp, ps_m, ps_a)
    """

    def __init__(self, t_max: float = 24.0, n_points: int = 30):
        super().__init__()
        self.ode_func = PBPKFunc()
        # Fewer time points for faster training; dense around peak
        t_dense = torch.linspace(0, 6.0, n_points * 2 // 3)
        t_sparse = torch.linspace(7.0, t_max, n_points // 3)
        self.register_buffer("t_eval", torch.cat([t_dense, t_sparse]))

    def forward(
        self,
        dose_mg: torch.Tensor,
        clint_hep: torch.Tensor,
        clint_gut: torch.Tensor,
        fup: torch.Tensor,
        ka: torch.Tensor,
        clr: torch.Tensor,
        kp: dict[str, torch.Tensor],
        ps_muscle: torch.Tensor,
        ps_adipose: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run PBPK simulation and return Cmax.

        All parameter tensors are (B,).

        Returns:
            cmax: (B,) Cmax in mg/L
            info: dict with C_venous time course, PK metrics
        """
        B = dose_mg.shape[0]
        device = dose_mg.device

        # Set drug parameters
        self.ode_func.set_params(clint_hep, clint_gut, fup, ka, clr, kp, ps_muscle, ps_adipose)

        # Initial conditions: all drug in GI lumen
        y0 = torch.zeros(B, N_STATES, device=device)
        y0[:, IDX_GI] = dose_mg

        # Solve ODE
        from torchdiffeq import odeint

        t_eval = self.t_eval.to(device)
        # y_sol shape: (T, B, 13)
        y_sol = odeint(
            self.ode_func,
            y0,
            t_eval,
            method="dopri5",
            rtol=1e-3,
            atol=1e-5,
            options={"max_num_steps": 500},
        )

        # Extract venous concentration: C_ven(t) = A_ven(t) / V_ven
        C_ven = y_sol[:, :, IDX_VEN] / V_VEN  # (T, B)

        # Cmax = max over time
        cmax, tmax_idx = C_ven.max(dim=0)  # (B,)
        tmax = t_eval[tmax_idx]  # (B,)

        # AUC via trapezoidal rule
        dt = t_eval[1:] - t_eval[:-1]  # (T-1,)
        C_mid = (C_ven[1:] + C_ven[:-1]) / 2  # (T-1, B)
        auc = (dt.unsqueeze(1) * C_mid).sum(dim=0)  # (B,)

        return cmax.clamp(min=1e-10), {
            "tmax": tmax,
            "auc": auc,
            "C_ven": C_ven,  # (T, B)
        }
