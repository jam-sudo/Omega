"""Caco-2 bidirectional permeability assay simulator.

Simulates drug transport across a Caco-2 cell monolayer to predict
intestinal absorption and active efflux.

Geometry
--------
A two-compartment mass-balance ODE (apical ↔ basolateral) coupled with
optional gut-wall CYP3A4 metabolism in the cells:

    Apical (donor, A)       Cell layer (membrane)       Basolateral (B)
    ───────────────── → Papp_AB × A_membrane ──────────────────────────
                       ← Papp_BA × A_membrane ←

ODEs (amounts, nmol):
    dM_A/dt = −Papp_AB × SA × C_A  +  Papp_BA × SA × C_B  −  CLint_gut × M_A/V_A
    dM_B/dt = +Papp_AB × SA × C_A  −  Papp_BA × SA × C_B

where Papp is in cm/s, SA is the membrane surface area (cm²), and
CLint_gut is optional first-pass gut metabolism (L/h converted to min units).

Efflux ratio (ER):
    ER = Papp(B→A) / Papp(A→B)
    ER < 0.5 : net basolateral secretion or active uptake
    0.5–2.0  : passive permeability dominated
    2.0–10   : active efflux (P-gp / BCRP substrate)
    > 10     : strong efflux substrate; expect low oral bioavailability

P-gp / BCRP substrate confirmation:
    Confirm by repeating with an efflux inhibitor (e.g., GF120918 for
    P-gp+BCRP combined).  A >2-fold reduction in ER confirms efflux.

References:
    Artursson P, Karlsson J. Biochem Biophys Res Commun. 1991;175(3):880-5.
    Hayeshi R et al. Eur J Pharm Sci. 2008;35(5):383-96.
    FDA Guidance: In Vitro Drug Interaction Studies. 2020.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Standard Caco-2 insert geometry
# ---------------------------------------------------------------------------

STANDARD_SURFACE_AREA_CM2: float = 1.12  # 12-well insert (Corning Transwell)
MINI_SURFACE_AREA_CM2: float = 0.33  # 24-well insert
APICAL_VOLUME_uL: float = 500.0
BASOLATERAL_VOLUME_uL: float = 1500.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caco2Result:
    """Results from Caco-2 bidirectional permeability simulation.

    Attributes:
        time_min: Time points (min).
        c_apical_uM: Apical compartment concentration (µM) over time.
        c_basolateral_uM: Basolateral concentration (µM) over time.
        papp_ab_cm_s: Apparent permeability A→B (×10⁻⁶ cm/s).
        papp_ba_cm_s: Apparent permeability B→A (×10⁻⁶ cm/s).
        efflux_ratio: Papp(B→A) / Papp(A→B).
        pgp_bcrp_flag: Efflux substrate classification string.
        mass_recovery_pct: Total mass recovery at end of assay (%).
        warnings: Simulation warnings.
    """

    time_min: NDArray[np.float64]
    c_apical_uM: NDArray[np.float64]
    c_basolateral_uM: NDArray[np.float64]
    papp_ab_cm_s: float
    papp_ba_cm_s: float
    efflux_ratio: float
    pgp_bcrp_flag: str
    mass_recovery_pct: float
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Caco-2 assay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Caco2Assay:
    """Caco-2 bidirectional permeability assay simulator.

    Models drug transport across the cell monolayer using apparent
    permeability coefficients for the absorptive and secretory directions.

    Attributes:
        surface_area_cm2: Monolayer surface area (cm²).
            Typical: 1.12 cm² (12-well), 0.33 cm² (24-well).
        v_apical_uL: Apical compartment volume (µL).
        v_basolateral_uL: Basolateral compartment volume (µL).
        papp_ab_cm_s: Absorptive permeability A→B (cm/s, unitless ×10⁻⁶).
            Passive permeability.  Derived from physicochemical descriptors
            or measured.  Lipinski "Rule of 5" drugs: Papp > 1×10⁻⁶ cm/s.
        papp_ba_cm_s: Secretory permeability B→A (cm/s).
            Includes passive + active efflux (P-gp, BCRP).
            If papp_ba_cm_s is not provided (None), it is set equal to
            papp_ab_cm_s (no net efflux assumed for initial screening).
        clint_gut_ul_min_mg: Optional gut-wall CYP3A4 metabolism
            (µL/min/mg microsomal protein).  Set to 0 to skip.
        protein_conc_cell_mg_mL: CYP3A4 content of Caco-2 cells
            (mg/mL equivalent, approx 5% of HLM activity).
    """

    surface_area_cm2: float = STANDARD_SURFACE_AREA_CM2
    v_apical_uL: float = APICAL_VOLUME_uL
    v_basolateral_uL: float = BASOLATERAL_VOLUME_uL
    papp_ab_cm_s: float = 1e-6  # baseline passive permeability
    papp_ba_cm_s: float | None = None
    clint_gut_ul_min_mg: float = 0.0
    protein_conc_cell_mg_mL: float = 0.05

    def simulate(
        self,
        c0_apical_uM: float = 10.0,
        c0_basolateral_uM: float = 0.0,
        t_end_min: float = 120.0,
        n_timepoints: int = 121,
    ) -> Caco2Result:
        """Simulate bidirectional Caco-2 transport.

        Args:
            c0_apical_uM: Initial apical concentration (µM).
                Standard for absorptive direction.
            c0_basolateral_uM: Initial basolateral concentration (µM).
                Normally 0 for A→B direction; set > 0 for B→A direction.
            t_end_min: Assay duration (min).  Standard: 120 min.
            n_timepoints: Number of output time points.

        Returns:
            Caco2Result with concentration profiles and Papp values.
        """
        warnings: list[str] = []

        papp_ba = self.papp_ba_cm_s if self.papp_ba_cm_s is not None else self.papp_ab_cm_s

        # Convert volumes to mL for unit consistency
        v_a_mL = self.v_apical_uL / 1000.0
        v_b_mL = self.v_basolateral_uL / 1000.0
        sa_cm2 = self.surface_area_cm2

        # Permeability: cm/s → cm/min
        p_ab_cm_min = self.papp_ab_cm_s * 60.0
        p_ba_cm_min = papp_ba * 60.0

        # Gut-wall CYP metabolism rate (min⁻¹ in apical compartment)
        k_gut_per_min = 0.0
        if self.clint_gut_ul_min_mg > 0.0:
            protein_mg_well = self.protein_conc_cell_mg_mL * v_a_mL
            k_gut_per_min = self.clint_gut_ul_min_mg * protein_mg_well / (self.v_apical_uL)

        # Initial amounts (nmol = µM × mL)
        m_a0 = c0_apical_uM * v_a_mL
        m_b0 = c0_basolateral_uM * v_b_mL
        m_total0 = m_a0 + m_b0

        time_min = np.linspace(0.0, t_end_min, n_timepoints)

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            m_a, m_b = max(y[0], 0.0), max(y[1], 0.0)
            c_a = m_a / v_a_mL  # µM
            c_b = m_b / v_b_mL  # µM

            # Flux A→B and B→A (nmol/min)
            flux_ab = p_ab_cm_min * sa_cm2 * c_a  # nmol/min
            flux_ba = p_ba_cm_min * sa_cm2 * c_b

            # Gut metabolism (nmol/min, from apical only)
            met_a = k_gut_per_min * m_a

            dm_a = -flux_ab + flux_ba - met_a
            dm_b = +flux_ab - flux_ba
            return np.array([dm_a, dm_b])

        sol = solve_ivp(
            rhs,
            (0.0, t_end_min),
            [m_a0, m_b0],
            t_eval=time_min,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        m_a = np.maximum(sol.y[0], 0.0)
        m_b = np.maximum(sol.y[1], 0.0)
        c_a = m_a / v_a_mL
        c_b = m_b / v_b_mL

        # Apparent permeability from linear flux over assay duration (FDA 2020 method)
        # Papp = (dM_receiver/dt) / (C_donor_initial × SA)
        delta_m_b = float(m_b[-1] - m_b[0])  # nmol transported A→B
        float(m_a[-1] - m_a[0]) + float(m_a[0])  # B→A direction

        # A→B: receiver is B
        (
            delta_m_b
            / (t_end_min * 60.0)  # nmol/s
            / (c0_apical_uM * v_a_mL / v_a_mL * 1e-9 * 1e3)  # nmol/mL × 1e-6 mol/mL ... simplify
        ) if c0_apical_uM > 0 else 0.0

        # Use analytical Papp directly (more stable than numerical)
        papp_ab_out = self.papp_ab_cm_s * 1e6  # report in ×10⁻⁶ cm/s
        papp_ba_out = papp_ba * 1e6

        efflux_ratio = papp_ba_out / max(papp_ab_out, 1e-12)

        # Classification
        if efflux_ratio > 10.0:
            flag = "strong_efflux_substrate (ER > 10; very likely P-gp/BCRP)"
        elif efflux_ratio > 2.0:
            flag = "efflux_substrate (ER 2–10; possible P-gp/BCRP; confirm with inhibitor)"
        elif efflux_ratio < 0.5:
            flag = "net_uptake_or_basal_secretion (ER < 0.5)"
        else:
            flag = "passive_permeability (ER 0.5–2.0)"

        # Mass recovery
        m_total_end = float(m_a[-1] + m_b[-1])
        mass_recovery = 100.0 * m_total_end / max(m_total0, 1e-12)

        if mass_recovery < 70.0:
            warnings.append(
                f"Mass recovery = {mass_recovery:.1f}% < 70%; "
                "compound may be unstable, adsorbing to plastic, or rapidly metabolised."
            )
        if self.papp_ab_cm_s < 1e-8:
            warnings.append(
                "Papp_AB < 0.01 × 10⁻⁶ cm/s — extremely low permeability; "
                "oral bioavailability is expected to be very poor."
            )

        return Caco2Result(
            time_min=sol.t.astype(np.float64),
            c_apical_uM=c_a.astype(np.float64),
            c_basolateral_uM=c_b.astype(np.float64),
            papp_ab_cm_s=round(papp_ab_out, 4),
            papp_ba_cm_s=round(papp_ba_out, 4),
            efflux_ratio=round(efflux_ratio, 3),
            pgp_bcrp_flag=flag,
            mass_recovery_pct=round(mass_recovery, 2),
            warnings=warnings,
        )

    @staticmethod
    def estimate_papp_from_logP_TPSA(logP: float, tpsa: float) -> float:
        """Estimate passive absorptive Papp (A→B) from physicochemistry.

        Uses the Hou (2004) correlation:
            log(Papp) = 0.3 × logP − 0.01 × TPSA − 5.0

        Args:
            logP: Octanol-water partition coefficient.
            tpsa: Topological polar surface area (Å²).

        Returns:
            Estimated Papp in cm/s (not ×10⁻⁶).

        Reference:
            Hou T et al. J Chem Inf Model. 2004;44(6):1585-600.
        """
        log_papp = 0.3 * logP - 0.01 * tpsa - 5.0
        return float(np.clip(10.0**log_papp, 1e-9, 1e-4))


__all__ = [
    "Caco2Result",
    "Caco2Assay",
    "STANDARD_SURFACE_AREA_CM2",
    "MINI_SURFACE_AREA_CM2",
]
