"""In vitro metabolic stability assay simulators.

Two standard assay systems are modelled:

1. **MicrosomesAssay** — Human liver microsome (HLM) incubation.
   Measures Phase 1 (CYP, FMO) intrinsic clearance via substrate
   depletion.  Scaling constants follow Hallifax & Houston (2006):
     MPPGL = 40 mg microsomal protein / g liver
     Liver weight = 1500 g (70 kg reference human)

2. **HepatocyteAssay** — Suspended hepatocyte incubation.
   Captures combined Phase 1 + Phase 2 (UGT, SULT) clearance.
   Scaling constant (Barter et al. 2007):
     HPGL = 120 × 10⁶ hepatocytes / g liver

Both assays use substrate-depletion kinetics (linear range, C0 << Km):

    dC_parent/dt = −k_dep × C_parent          [first-order loss]
    dC_met/dt   =  f_met × k_dep × C_parent − k_met_elim × C_met

where k_dep (h⁻¹) is back-calculated from CLint and the incubation
system volume.

References:
    Hallifax D, Houston JB. Drug Metab Dispos. 2006;34(5):724-32.
    Barter ZE et al. Curr Drug Metab. 2007;8(1):33-45.
    Obach RS. Drug Metab Dispos. 1999;27(11):1350-9.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Scaling constants
# ---------------------------------------------------------------------------

MPPGL: float = 40.0  # mg microsomal protein / g liver
HPGL: float = 120e6  # hepatocytes / g liver
LIVER_WEIGHT_G: float = 1500.0  # reference human liver (70 kg)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssayResult:
    """Output from a metabolic stability assay simulation.

    Attributes:
        time_min: Incubation time points (minutes).
        c_parent: Parent compound concentration (µM) over time.
        c_metabolite: Primary metabolite concentration (µM) over time.
            All zeros if metabolite tracking is disabled.
        k_dep_per_h: First-order depletion rate constant (h⁻¹).
        t_half_min: Apparent half-life (min) from depletion curve.
        clint_ul_min_per_system: Intrinsic clearance in assay units.
            µL/min/mg protein (microsomes) or µL/min/10⁶ cells (hepatocytes).
        clint_liver_L_per_h: CLint scaled to whole liver (L/h).
        assay_type: "microsomes" or "hepatocytes".
        warnings: List of any simulation warnings.
    """

    time_min: NDArray[np.float64]
    c_parent: NDArray[np.float64]
    c_metabolite: NDArray[np.float64]
    k_dep_per_h: float
    t_half_min: float
    clint_ul_min_per_system: float
    clint_liver_L_per_h: float
    assay_type: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Microsomes assay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MicrosomesAssay:
    """Human liver microsome (HLM) metabolic stability simulator.

    Simulates the substrate-depletion assay in a microsomal incubation
    well and back-calculates CLint from the depletion half-life.

    Attributes:
        protein_conc_mg_mL: Microsomal protein concentration in well
            (mg/mL).  Typical range: 0.1–1.0 mg/mL.
        volume_uL: Incubation volume (µL).  Typical: 100–500 µL.
        f_unbound_mic: Fraction unbound in microsomal system (fu_mic).
            Defaults to 1.0 (no non-specific microsomal binding).
            Binding correction: CLint_corrected = CLint_nominal / fu_mic.
            Can be measured (rapid equilibrium dialysis) or estimated
            from logP: fu_mic ≈ 1 / (1 + 0.0072 × P_ow)
            (Austin et al. 2002, Drug Metab Dispos 30:1497-1503).
        f_metabolite: Fraction of parent converted to primary metabolite
            (0–1).  Set to 0 to disable metabolite tracking.
        k_met_elim_per_h: First-order elimination rate of the primary
            metabolite from the incubation (h⁻¹).  Defaults to 0 (stable).
    """

    protein_conc_mg_mL: float = 0.5
    volume_uL: float = 200.0
    f_unbound_mic: float = 1.0
    f_metabolite: float = 0.0
    k_met_elim_per_h: float = 0.0

    def simulate(
        self,
        clint_ul_min_mg: float,
        c0_uM: float = 1.0,
        t_end_min: float = 60.0,
        n_timepoints: int = 61,
    ) -> AssayResult:
        """Simulate microsomal substrate depletion.

        Args:
            clint_ul_min_mg: Nominal intrinsic clearance (µL/min/mg protein).
                This is the value measured at the stated fu_mic = 1.  The
                assay simulator internally applies the fu_mic correction.
            c0_uM: Initial parent compound concentration (µM).
                Keep ≪ Km (< 1 µM for most CYP substrates) to stay in
                linear range.
            t_end_min: Duration of incubation (min).
            n_timepoints: Number of output time points.

        Returns:
            AssayResult with parent/metabolite time courses and CLint.
        """
        warnings: list[str] = []

        # Binding-corrected CLint
        clint_corr = clint_ul_min_mg / max(self.f_unbound_mic, 1e-6)

        # First-order depletion rate in the incubation well (min⁻¹)
        # k_dep = CLint_corrected × protein_amount / volume
        #       = CLint_corrected [µL/min/mg] × protein [mg] / volume [µL]
        protein_mg = self.protein_conc_mg_mL * (self.volume_uL / 1000.0)
        k_dep_per_min = clint_corr * protein_mg / self.volume_uL
        k_dep_per_h = k_dep_per_min * 60.0

        if c0_uM > 10.0:
            warnings.append(
                f"c0 = {c0_uM} µM may exceed linear range (Km typically 1–10 µM); "
                "CLint back-calculation assumes first-order kinetics."
            )

        time_min = np.linspace(0.0, t_end_min, n_timepoints)

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            c_p, c_m = y[0], y[1]
            loss = k_dep_per_min * max(c_p, 0.0)
            dc_p = -loss
            dc_m = self.f_metabolite * loss - (self.k_met_elim_per_h / 60.0) * max(c_m, 0.0)
            return np.array([dc_p, dc_m])

        sol = solve_ivp(
            rhs,
            (0.0, t_end_min),
            [c0_uM, 0.0],
            t_eval=time_min,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        c_parent = np.maximum(sol.y[0], 0.0)
        c_metabolite = np.maximum(sol.y[1], 0.0)

        t_half_min = np.log(2.0) / max(k_dep_per_min, 1e-12)

        # Back-calculate CLint from the well (µL/min/mg protein)
        clint_back = k_dep_per_min * self.volume_uL / max(protein_mg, 1e-12)

        # Scale to whole liver (L/h)
        clint_liver_L_per_h = (
            clint_corr  # µL/min/mg
            * MPPGL  # mg protein/g liver
            * LIVER_WEIGHT_G  # g liver
            / 1e6  # µL → L
            * 60.0  # min → h
        )

        return AssayResult(
            time_min=sol.t.astype(np.float64),
            c_parent=c_parent.astype(np.float64),
            c_metabolite=c_metabolite.astype(np.float64),
            k_dep_per_h=k_dep_per_h,
            t_half_min=round(t_half_min, 2),
            clint_ul_min_per_system=round(clint_back, 4),
            clint_liver_L_per_h=round(clint_liver_L_per_h, 4),
            assay_type="microsomes",
            warnings=warnings,
        )

    @staticmethod
    def estimate_fu_mic(logP: float) -> float:
        """Estimate microsomal binding (fu_mic) from logP.

        Uses the Austin (2002) regression:
            fu_mic = 1 / (1 + 0.0072 × 10^logP)

        Args:
            logP: Octanol-water partition coefficient.

        Returns:
            fu_mic in range (0, 1].
        """
        return float(1.0 / (1.0 + 0.0072 * 10.0**logP))


# ---------------------------------------------------------------------------
# Hepatocyte assay
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HepatocyteAssay:
    """Suspended hepatocyte metabolic stability simulator.

    Captures combined Phase 1 + Phase 2 (UGT, SULT) clearance.
    Cell viability and non-specific binding to cells are accounted for.

    Attributes:
        cell_density_million_per_mL: Hepatocyte density (10⁶ cells/mL).
            Typical: 0.5–2.0 × 10⁶ cells/mL.
        volume_uL: Incubation volume (µL).
        viability_fraction: Fraction of viable cells (0–1).
            Recommended: use only cells with > 80% viability.
        fu_cell: Fraction unbound in cell incubation (0–1).
            Accounts for non-specific binding to cell membranes.
            If not measured, set to 1.0 (conservative).
        f_metabolite: Fraction converted to primary metabolite.
        k_met_elim_per_h: First-order metabolite loss rate (h⁻¹).
    """

    cell_density_million_per_mL: float = 1.0
    volume_uL: float = 200.0
    viability_fraction: float = 0.9
    fu_cell: float = 1.0
    f_metabolite: float = 0.0
    k_met_elim_per_h: float = 0.0

    def simulate(
        self,
        clint_ul_min_million: float,
        c0_uM: float = 1.0,
        t_end_min: float = 120.0,
        n_timepoints: int = 121,
    ) -> AssayResult:
        """Simulate hepatocyte substrate depletion.

        Args:
            clint_ul_min_million: Intrinsic clearance (µL/min/10⁶ cells).
                Total clearance including Phase 1 + Phase 2.
            c0_uM: Initial parent compound concentration (µM).
            t_end_min: Incubation duration (min).
            n_timepoints: Number of output time points.

        Returns:
            AssayResult with depletion kinetics and whole-liver CLint.
        """
        warnings: list[str] = []

        # Effective CLint accounting for cell viability and binding
        clint_eff = clint_ul_min_million / max(self.fu_cell, 1e-6)
        clint_viable = clint_eff * self.viability_fraction

        if self.viability_fraction < 0.8:
            warnings.append(
                f"Cell viability {self.viability_fraction:.0%} < 80%; "
                "CLint estimate may be underestimated."
            )

        # Effective cell count in well
        cell_million = self.cell_density_million_per_mL * (self.volume_uL / 1000.0)
        k_dep_per_min = clint_viable * cell_million / self.volume_uL
        k_dep_per_h = k_dep_per_min * 60.0

        time_min = np.linspace(0.0, t_end_min, n_timepoints)

        def rhs(t: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
            c_p, c_m = y[0], y[1]
            loss = k_dep_per_min * max(c_p, 0.0)
            dc_p = -loss
            dc_m = self.f_metabolite * loss - (self.k_met_elim_per_h / 60.0) * max(c_m, 0.0)
            return np.array([dc_p, dc_m])

        sol = solve_ivp(
            rhs,
            (0.0, t_end_min),
            [c0_uM, 0.0],
            t_eval=time_min,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10,
        )

        c_parent = np.maximum(sol.y[0], 0.0)
        c_metabolite = np.maximum(sol.y[1], 0.0)

        t_half_min = np.log(2.0) / max(k_dep_per_min, 1e-12)
        clint_back = k_dep_per_min * self.volume_uL / max(cell_million, 1e-12)

        # Scale to whole liver (Barter 2007)
        clint_liver_L_per_h = (
            clint_viable  # µL/min/10⁶ cells
            * (HPGL / 1e6)  # 10⁶ cells/g liver
            * LIVER_WEIGHT_G  # g liver
            / 1e6  # µL → L
            * 60.0  # min → h
        )

        return AssayResult(
            time_min=sol.t.astype(np.float64),
            c_parent=c_parent.astype(np.float64),
            c_metabolite=c_metabolite.astype(np.float64),
            k_dep_per_h=k_dep_per_h,
            t_half_min=round(t_half_min, 2),
            clint_ul_min_per_system=round(clint_back, 4),
            clint_liver_L_per_h=round(clint_liver_L_per_h, 4),
            assay_type="hepatocytes",
            warnings=warnings,
        )


__all__ = [
    "AssayResult",
    "MicrosomesAssay",
    "HepatocyteAssay",
    "MPPGL",
    "HPGL",
    "LIVER_WEIGHT_G",
]
