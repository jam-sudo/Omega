"""WholeBodyPBPK — 35-state ODE engine for 15-organ whole-body PBPK simulation.

State vector layout (35 states):
  [0]  venous_blood
  [1]  arterial_blood
  [2]  lung
  [3]  brain
  [4]  heart
  [5]  kidney
  [6]  liver
  [7]  spleen
  [8]  gut_wall
  [9]  pancreas
  [10] thymus
  [11] reproductive
  [12] rest
  [13] adipose_vasc       (permeability-limited)
  [14] muscle_vasc        (permeability-limited)
  [15] bone_vasc          (permeability-limited)
  [16] skin_vasc          (permeability-limited)
  [17] adipose_extra      (permeability-limited)
  [18] muscle_extra       (permeability-limited)
  [19] bone_extra         (permeability-limited)
  [20] skin_extra         (permeability-limited)
  [21] stomach_lumen      (ACAT)
  [22] duodenum_lumen     (ACAT)
  [23] jejunum1_lumen     (ACAT)
  [24] jejunum2_lumen     (ACAT)
  [25] ileum1_lumen       (ACAT)
  [26] ileum2_lumen       (ACAT)
  [27] ileum3_lumen       (ACAT)
  [28] colon_lumen        (ACAT)
  [29] portal_vein
  [30] metabolized_hepatic
  [31] excreted_renal
  [32] metabolized_gut
  [33] excreted_fecal
  [34] sc_depot           (subcutaneous absorption depot)

SC absorption:
  dA_SC/dt = -ka_sc * A_SC
  Absorbed drug enters venous blood directly (bypasses first-pass).
  Effective absorbed rate = f_sc * ka_sc * A_SC

Mass balance (IV): dose = sum(all states) at all times.
Solver: scipy solve_ivp, method='LSODA', rtol=1e-8, atol=1e-10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from omega_pbpk._compat import np_trapz
from omega_pbpk.core.organ import Organ
from omega_pbpk.drugs.drug import Drug

logger = logging.getLogger(__name__)

# Phase 3a.2 feature flag: use villous blood flow (~6% CO) for Fg calculation
# instead of full mesenteric/intestinal flow (~15% CO). Villous flow is the
# physiologically correct flow for the well-stirred gut wall model (Yang 2007).
_USE_VILLOUS_FLOW = False  # disabled (must be enabled together with ENABLE_GUT_WALL_FIX)

N_STATES = 35

# State indices
IDX_VEN = 0
IDX_ART = 1
IDX_LUNG = 2
IDX_BRAIN = 3
IDX_HEART = 4
IDX_KIDNEY = 5
IDX_LIVER = 6
IDX_SPLEEN = 7
IDX_GUT_WALL = 8
IDX_PANCREAS = 9
IDX_THYMUS = 10
IDX_REPRO = 11
IDX_REST = 12
IDX_ADIPOSE_V = 13
IDX_MUSCLE_V = 14
IDX_BONE_V = 15
IDX_SKIN_V = 16
IDX_ADIPOSE_E = 17
IDX_MUSCLE_E = 18
IDX_BONE_E = 19
IDX_SKIN_E = 20
IDX_STOMACH = 21
IDX_DUODENUM = 22
IDX_JEJUNUM1 = 23
IDX_JEJUNUM2 = 24
IDX_ILEUM1 = 25
IDX_ILEUM2 = 26
IDX_ILEUM3 = 27
IDX_COLON = 28
IDX_PORTAL = 29
IDX_MET_HEPATIC = 30
IDX_EXC_RENAL = 31
IDX_MET_GUT = 32
IDX_EXC_FECAL = 33
IDX_SC_DEPOT = 34  # Subcutaneous absorption depot

# ACAT segment indices and transit rates (h⁻¹)
ACAT_INDICES = [
    IDX_STOMACH,
    IDX_DUODENUM,
    IDX_JEJUNUM1,
    IDX_JEJUNUM2,
    IDX_ILEUM1,
    IDX_ILEUM2,
    IDX_ILEUM3,
    IDX_COLON,
]
ACAT_TRANSIT_TIMES_H = [0.25, 0.26, 0.475, 0.475, 0.68, 0.68, 0.68, 13.5]
ACAT_KA_FRACTIONS = [0.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.3, 0.05]

# Perfusion-limited organ names and their state indices
PERFUSION_LIMITED_ORGANS = [
    ("lung", IDX_LUNG),
    ("brain", IDX_BRAIN),
    ("heart", IDX_HEART),
    ("kidney", IDX_KIDNEY),
    ("liver", IDX_LIVER),
    ("spleen", IDX_SPLEEN),
    ("gut_wall", IDX_GUT_WALL),
    ("pancreas", IDX_PANCREAS),
    ("thymus", IDX_THYMUS),
    ("reproductive", IDX_REPRO),
    ("rest", IDX_REST),
]

# Permeability-limited organ names and their state indices (vascular, extravascular)
PERMEABILITY_LIMITED_ORGANS = [
    ("adipose", IDX_ADIPOSE_V, IDX_ADIPOSE_E),
    ("muscle", IDX_MUSCLE_V, IDX_MUSCLE_E),
    ("bone", IDX_BONE_V, IDX_BONE_E),
    ("skin", IDX_SKIN_V, IDX_SKIN_E),
]

# Organs that drain into portal vein (→ liver)
PORTAL_ORGANS = {"spleen", "gut_wall", "pancreas"}


@dataclass
class SimulationResult:
    """Container for simulation output."""

    time_h: NDArray[np.floating[Any]]
    amounts: NDArray[np.floating[Any]]  # shape (n_time, 35)
    drug: Drug
    organs: dict[str, Organ] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        return len(self.time_h)

    def plasma_concentration(self) -> NDArray[np.floating[Any]]:
        """Plasma concentration (mg/L) from venous blood."""
        v_ven = self.organs["venous_blood"].volume_L
        c_blood = self.amounts[:, IDX_VEN] / max(v_ven, 1e-12)
        return c_blood / max(self.drug.rbp, 1e-12)

    def pk_summary(self) -> dict[str, float]:
        """Compute standard PK parameters from plasma concentration-time profile."""
        cp = self.plasma_concentration()
        t = self.time_h

        cmax = float(np.max(cp))
        tmax = float(t[np.argmax(cp)])
        auc = float(np_trapz(cp, t))

        # Terminal half-life: linear regression of log(Cp) vs time over last 50%
        # R² < 0.90 warns that profile may not have entered terminal phase
        # (common when t_end_h < 2 × t½, e.g. warfarin t½ 40 h simulated to 24 h).
        n = len(t)
        start = max(n // 2, 1)
        cp_term = cp[start:]
        t_term = t[start:]
        mask = cp_term > cmax * 0.01
        if np.sum(mask) >= 3:
            log_cp = np.log(np.maximum(cp_term[mask], 1e-30))
            t_fit = t_term[mask]
            # Linear regression of log(Cp) vs time
            coeffs = np.polyfit(t_fit, log_cp, 1)
            ke = -coeffs[0]
            half_life = 0.693 / max(ke, 1e-12) if ke > 0 else float("inf")
            # Quality checks — two failure modes:
            # (1) Poor log-linear fit: bi-exponential visible in window (R² < 0.90)
            # (2) t½ >> simulation time: profile looks mono-exp but wrong phase
            if ke > 0:
                residuals = log_cp - np.polyval(coeffs, t_fit)
                ss_res = float(np.sum(residuals**2))
                ss_tot = float(np.sum((log_cp - float(np.mean(log_cp))) ** 2))
                r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
                t_end = float(t[-1])
                if r2 < 0.90:
                    logger.warning(
                        "Terminal t½ fit R²=%.3f < 0.90 at t_end=%.1f h "
                        "(estimated t½=%.1f h). Profile may not have reached "
                        "terminal phase — extend t_end_h for reliable t½.",
                        r2,
                        t_end,
                        half_life,
                    )
                elif half_life > t_end:
                    logger.warning(
                        "Estimated t½ (%.1f h) exceeds simulation duration (%.1f h). "
                        "Terminal phase not captured — extend t_end_h for accurate t½.",
                        half_life,
                        t_end,
                    )
        else:
            half_life = float("inf")

        cl = self.drug.dose_mg / max(auc, 1e-12) if auc > 0 else 0.0
        vss = cl * half_life / 0.693 if half_life < float("inf") else 0.0

        return {
            "Cmax_mg_L": round(cmax, 6),
            "Tmax_h": round(tmax, 4),
            "AUC_mg_h_L": round(auc, 4),
            "half_life_h": round(half_life, 4) if half_life < 1e6 else float("inf"),
            "CL_L_h": round(cl, 6),
            "Vss_L": round(vss, 4),
            "dose_mg": self.drug.dose_mg,
            "route": self.drug.route,
        }

    def mass_balance(self) -> NDArray[np.floating[Any]]:
        """Total drug mass at each time point (should equal dose)."""
        return np.sum(self.amounts, axis=1)


@dataclass
class DDIInhibitor:
    """Drug-drug interaction inhibitor specification."""

    name: str
    ki_uM: float
    concentration_uM: float
    target_enzyme: str = "CYP3A4"
    mechanism: str = "competitive"  # competitive, mbi, induction
    kinact_per_h: float = 0.0  # For MBI
    kdeg_per_h: float = 0.02  # Enzyme degradation rate
    fold_induction: float = 1.0  # For induction


class WholeBodyPBPK:
    """35-state ODE-based whole-body PBPK model.

    Usage:
        model = WholeBodyPBPK(drug, body_weight=70.0)
        model.setup_oral(dose_mg=7.5)
        result = model.simulate(t_end_h=24.0)
    """

    def __init__(
        self, drug: Drug, body_weight: float = 70.0, age_years: float | None = None
    ) -> None:
        self.drug = drug
        self.body_weight = body_weight
        self.age_years = age_years
        self.organs: dict[str, Organ] = {}
        self.inhibitors: list[DDIInhibitor] = []
        self._y0 = np.zeros(N_STATES)
        self._ontogeny_factors = self._compute_ontogeny_factors()
        self._build_organs()

    def _compute_ontogeny_factors(self) -> dict[str, float]:
        """Compute pediatric ontogeny scaling factors.

        Returns all-1.0 for adults (age_years is None or >= 18).
        For pediatric subjects, returns enzyme and GFR maturation factors.
        """
        if self.age_years is None or self.age_years >= 18.0:
            return {"cyp3a4": 1.0, "cyp2d6": 1.0, "cyp1a2": 1.0, "gfr": 1.0}
        from omega_pbpk.clinical.ontogeny import get_pediatric_scaling

        return get_pediatric_scaling(self.age_years, self.body_weight)

    def _build_organs(self) -> None:
        """Construct organ compartments from ICRP reference man physiology.

        Scaled linearly by body_weight/70.
        """
        scale = self.body_weight / 70.0
        if self.age_years is not None and self.age_years < 18.0:
            co = 390.0 * self._ontogeny_factors["cardiac_output"]
        else:
            co = 390.0 * scale  # Cardiac output (L/h)

        # Blood pools
        self.organs["venous_blood"] = Organ(
            "venous_blood", volume_L=3.7 * scale, blood_flow_L_per_h=co
        )
        self.organs["arterial_blood"] = Organ(
            "arterial_blood", volume_L=1.5 * scale, blood_flow_L_per_h=co
        )

        # Perfusion-limited organs: (name, volume_L @70kg, %CO)
        # Flow fractions (non-lung) MUST sum to 1.0 for mass balance
        perf_data: list[tuple[str, float, float]] = [
            ("lung", 0.50, 1.0),
            ("brain", 1.45, 0.12),
            ("heart", 0.33, 0.04),
            ("kidney", 0.31, 0.19),
            ("liver", 1.80, 0.065),  # hepatic artery only
            ("spleen", 0.15, 0.03),
            ("gut_wall", 1.03, 0.15),  # intestinal flow
            ("pancreas", 0.10, 0.01),
            ("thymus", 0.02, 0.002),
            ("reproductive", 0.04, 0.002),
            ("rest", 2.50, 0.069),  # Balance: 1.0 - sum(others)
        ]

        for name, vol, frac_co in perf_data:
            kp = self.drug.default_kp(name)
            self.organs[name] = Organ(
                name=name,
                volume_L=vol * scale,
                blood_flow_L_per_h=co * frac_co,
                kp=kp,
            )

        # Permeability-limited organs: (name, volume_L @70kg, %CO)
        perm_data: list[tuple[str, float, float]] = [
            ("adipose", 14.5, 0.052),
            ("muscle", 28.0, 0.17),
            ("bone", 4.86, 0.05),
            ("skin", 3.30, 0.05),
        ]

        for name, vol, frac_co in perm_data:
            pl = self.drug.permeability_limited.get(name, {})
            kp = pl.get("kp", self.drug.default_kp(name))
            ps = pl.get("ps", 10.0)
            self.organs[name] = Organ(
                name=name,
                volume_L=vol * scale,
                blood_flow_L_per_h=co * frac_co,
                kp=kp,
                is_permeability_limited=True,
                ps_L_per_h=ps * scale,
            )

        # Portal vein (small mixing compartment)
        self.organs["portal_vein"] = Organ(
            "portal_vein", volume_L=0.05 * scale, blood_flow_L_per_h=0.0
        )

        self.cardiac_output = co

    def _compute_fg(self) -> float:
        """Gut wall first-pass fraction escaping (Fg) using Qgut model (Yang 2007).

        Fg = Qgut / (Qgut + fup × CLint_gut)

        When _USE_VILLOUS_FLOW is enabled, uses villous blood flow (~6% CO,
        ~23 L/h for 70kg) instead of full intestinal flow (~15% CO, ~58 L/h).
        Villous flow is the physiologically correct perfusion for the enterocyte
        well-stirred model.
        """
        if _USE_VILLOUS_FLOW:
            q_gut = 0.06 * self.cardiac_output  # villous flow ~6% CO
        else:
            q_gut = self.organs["gut_wall"].blood_flow_L_per_h
        cl_gut = self.drug.gut_clint_scaled_L_per_h
        if cl_gut <= 0:
            return 1.0
        return q_gut / (q_gut + self.drug.fup * cl_gut)

    def _clint_effective(self) -> float:
        """Effective hepatic CLint considering pediatric maturation and DDI inhibitors (L/h)."""
        base_clint = self.drug.clint_scaled_L_per_h

        # Apply pediatric enzyme maturation
        if self.age_years is not None and self.age_years < 18.0:
            fm = self.drug.fm
            if fm:
                enzyme_map = {"CYP3A4": "cyp3a4", "CYP2D6": "cyp2d6", "CYP1A2": "cyp1a2"}
                weighted_mat = 0.0
                for enz, frac in fm.items():
                    onto_key = enzyme_map.get(enz)
                    mat = self._ontogeny_factors.get(onto_key, 1.0) if onto_key else 1.0
                    weighted_mat += frac * mat
                base_clint *= weighted_mat
            else:
                base_clint *= self._ontogeny_factors.get("cyp3a4", 1.0)

        if not self.inhibitors:
            return max(base_clint, 0.0)

        # Apply inhibition for each enzyme
        for inh in self.inhibitors:
            fm_target = self.drug.fm.get(inh.target_enzyme, 0.0)
            if fm_target <= 0:
                continue

            # Guard: ki <= 0 is non-physical; skip inhibition (no effect)
            if inh.ki_uM <= 0:
                logger.warning(
                    "DDI inhibitor '%s' has non-positive ki_uM=%.4g; "
                    "skipping inhibition (treating as no effect).",
                    inh.name,
                    inh.ki_uM,
                )
                continue

            if inh.mechanism == "competitive":
                # Competitive: CLint_eff = CLint × (1 - fm × (1 - 1/(1 + [I]/Ki)))
                inhibition_factor = 1.0 / (1.0 + inh.concentration_uM / inh.ki_uM)
                base_clint *= 1.0 - fm_target * (1.0 - inhibition_factor)

            elif inh.mechanism == "mbi":
                # Mechanism-based inactivation: CLint_eff adjusted
                numerator = inh.kinact_per_h * inh.concentration_uM
                denominator = (
                    inh.kdeg_per_h * (inh.ki_uM + inh.concentration_uM)
                    + inh.kinact_per_h * inh.concentration_uM
                )
                if denominator > 0:
                    frac_inact = numerator / denominator
                    base_clint *= 1.0 - fm_target * frac_inact

            elif inh.mechanism == "induction":
                # Induction: CLint_eff = CLint × (1 + fm × (fold_induction - 1))
                base_clint *= 1.0 + fm_target * (inh.fold_induction - 1.0)

        return max(base_clint, 0.0)

    def setup_iv(self, dose_mg: float) -> None:
        """Set up intravenous bolus administration."""
        self._y0 = np.zeros(N_STATES)
        self._y0[IDX_VEN] = dose_mg  # Inject into venous blood
        self.drug = Drug(**{**self.drug.__dict__, "dose_mg": dose_mg, "route": "iv"})

    def setup_oral(self, dose_mg: float) -> None:
        """Set up oral administration into stomach lumen."""
        self._y0 = np.zeros(N_STATES)
        self._y0[IDX_STOMACH] = dose_mg  # Drug in stomach lumen
        self.drug = Drug(**{**self.drug.__dict__, "dose_mg": dose_mg, "route": "oral"})

    def setup_sc(self, dose_mg: float) -> None:
        """Set up subcutaneous (SC) administration.

        Drug is placed in an SC depot and absorbed via first-order kinetics
        directly into systemic venous blood (no first-pass effect).

        dA_SC/dt = -ka_sc * A_SC
        The rate entering venous blood = f_sc * ka_sc * A_SC

        Args:
            dose_mg: Administered SC dose (mg).
        """
        self._y0 = np.zeros(N_STATES)
        self._y0[IDX_SC_DEPOT] = dose_mg  # Drug in SC depot
        self.drug = Drug(**{**self.drug.__dict__, "dose_mg": dose_mg, "route": "sc"})

    def add_inhibitor(self, inhibitor: DDIInhibitor) -> None:
        """Add a DDI inhibitor to the simulation."""
        self.inhibitors.append(inhibitor)

    def _rhs(self, t: float, y: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Right-hand side of the 35-state ODE system.

        This is the heart of the PBPK model. All pharmacokinetics emerge from
        these differential equations.
        """
        dydt = np.zeros(N_STATES)

        fup = self.drug.fup
        rbp = max(self.drug.rbp, 1e-12)
        co = self.cardiac_output

        # Blood pool volumes
        v_ven = max(self.organs["venous_blood"].volume_L, 1e-12)
        v_art = max(self.organs["arterial_blood"].volume_L, 1e-12)

        # Blood concentrations
        c_ven = y[IDX_VEN] / v_ven
        c_art = y[IDX_ART] / v_art

        # Lung — feeds arterial blood
        o_lung = self.organs["lung"]
        v_lung = max(o_lung.volume_L, 1e-12)
        kp_lung = max(o_lung.kp, 1e-12)
        c_lung_out = y[IDX_LUNG] * rbp / (v_lung * kp_lung)
        dydt[IDX_LUNG] = co * c_ven - co * c_lung_out

        # Arterial blood: receives from lung, distributes to all organs
        dydt[IDX_ART] = co * c_lung_out - co * c_art

        # Venous blood: collects from all organs (except lung)
        venous_inflow = 0.0

        # --- Perfusion-limited organs (excluding lung) ---
        portal_inflow = 0.0

        for name, idx in PERFUSION_LIMITED_ORGANS:
            if name == "lung":
                continue
            org = self.organs[name]
            v_t = max(org.volume_L, 1e-12)
            kp = max(org.kp, 1e-12)
            q = org.blood_flow_L_per_h

            # Venous concentration leaving tissue
            c_out = y[idx] * rbp / (v_t * kp)

            if name == "liver":
                # Liver receives from hepatic artery + portal vein
                q_ha = q
                v_pv = max(self.organs["portal_vein"].volume_L, 1e-12)
                c_pv = y[IDX_PORTAL] / v_pv
                q_portal = sum(self.organs[pn].blood_flow_L_per_h for pn in PORTAL_ORGANS)
                q_total = q_ha + q_portal

                # Well-stirred model: CLh = (Q × fup × CLint) / (Q + fup × CLint)
                clint_eff = self._clint_effective()
                clh = (q_total * fup * clint_eff) / max(q_total + fup * clint_eff, 1e-12)
                # CLh from well-stirred model is on a total concentration basis
                # Applied to mixed input blood concentration (HA + PV)
                c_liver_in = (q_ha * c_art + q_portal * c_pv) / max(q_total, 1e-12)
                met_rate = clh * c_liver_in

                dydt[idx] = q_ha * c_art + q_portal * c_pv - q_total * c_out - met_rate
                dydt[IDX_MET_HEPATIC] += met_rate
                venous_inflow += q_total * c_out

            elif name == "kidney":
                # CLr defined on total plasma concentration basis
                c_plasma_out = y[idx] / (v_t * kp)
                gfr_factor = self._ontogeny_factors.get("gfr", 1.0)
                renal_cl_rate = self.drug.clr_L_per_h * gfr_factor * c_plasma_out

                dydt[idx] = q * c_art - q * c_out - renal_cl_rate
                dydt[IDX_EXC_RENAL] += renal_cl_rate
                venous_inflow += q * c_out

            elif name in PORTAL_ORGANS:
                # Portal organs drain into portal vein, not venous blood
                if name == "gut_wall":
                    # Gut wall receives absorbed drug from ACAT + arterial blood
                    # Gut metabolism (Fg)
                    cl_gut = self.drug.gut_clint_scaled_L_per_h
                    gut_met_rate = cl_gut * fup * y[idx] / (v_t * kp)

                    dydt[idx] = q * c_art - q * c_out - gut_met_rate
                    dydt[IDX_MET_GUT] += gut_met_rate
                else:
                    dydt[idx] = q * c_art - q * c_out
                portal_inflow += q * c_out
            else:
                dydt[idx] = q * c_art - q * c_out
                venous_inflow += q * c_out

        # --- Permeability-limited organs ---
        for name, idx_v, idx_e in PERMEABILITY_LIMITED_ORGANS:
            org = self.organs[name]
            q = org.blood_flow_L_per_h
            ps = org.ps_L_per_h or 10.0
            kp = max(org.kp, 1e-12)
            v_vasc = max(org.vascular_volume_L, 1e-12)
            v_extra = max(org.extravascular_volume_L, 1e-12)

            c_vasc = y[idx_v] / v_vasc
            c_extra = y[idx_e] / v_extra

            # Unbound concentrations for PS transfer
            cu_vasc = fup * c_vasc / rbp
            cu_extra = fup * c_extra / kp

            # Vascular: arterial in - venous out - PS transfer
            c_vasc_out = c_vasc  # blood concentration leaving vascular space
            dydt[idx_v] = q * c_art - q * c_vasc_out - ps * (cu_vasc - cu_extra)
            dydt[idx_e] = ps * (cu_vasc - cu_extra)

            venous_inflow += q * c_vasc_out

        # --- Portal vein ---
        v_pv = max(self.organs["portal_vein"].volume_L, 1e-12)
        q_portal_total = sum(self.organs[pn].blood_flow_L_per_h for pn in PORTAL_ORGANS)
        dydt[IDX_PORTAL] = portal_inflow - q_portal_total * (y[IDX_PORTAL] / v_pv)

        # --- ACAT absorption model ---
        # Process all segments: use += to preserve transit_in from previous segment
        for i, (seg_idx, transit_time, ka_frac) in enumerate(
            zip(ACAT_INDICES, ACAT_TRANSIT_TIMES_H, ACAT_KA_FRACTIONS, strict=False)
        ):
            kt = 1.0 / max(transit_time, 1e-6)  # Transit rate

            # Absorption rate from this segment
            # ka = 2 × Peff / radius × scaling
            peff = self.drug.peff  # ×10⁻⁴ cm/s
            ka_base = 2.0 * peff * 3600.0 * 1e-4 / max(self.drug.particle_radius_um * 1e-4, 1e-12)
            ka = (
                ka_base * ka_frac * 0.0004
            )  # recalibrated on 1,020 MMPK drugs (Optuna E2E, was 0.0007)

            absorption = ka * y[seg_idx]
            transit_out = kt * y[seg_idx]

            # Loss from this segment (absorption + transit out)
            dydt[seg_idx] += -absorption - transit_out

            # Transit to next segment or fecal excretion
            if i < len(ACAT_INDICES) - 1:
                dydt[ACAT_INDICES[i + 1]] += transit_out
            else:
                dydt[IDX_EXC_FECAL] += transit_out

            # Drug absorbed goes to gut wall
            dydt[IDX_GUT_WALL] += absorption

        # --- SC depot absorption ---
        # First-order absorption from SC depot into venous blood.
        # dA_SC/dt = -ka_sc * A_SC
        # Rate into venous blood = f_sc * ka_sc * A_SC (bioavailability-corrected)
        ka_sc = self.drug.ka_sc
        f_sc = self.drug.f_sc
        sc_amount = y[IDX_SC_DEPOT]
        sc_absorption_rate = ka_sc * sc_amount
        dydt[IDX_SC_DEPOT] = -sc_absorption_rate
        # Absorbed fraction enters venous blood; lost fraction (1-f_sc) is
        # pre-systemic degradation at the injection site — tracked in the
        # hepatic metabolized sink to preserve mass balance.
        venous_inflow += f_sc * sc_absorption_rate
        dydt[IDX_MET_HEPATIC] += (1.0 - f_sc) * sc_absorption_rate

        # --- Venous blood ---
        dydt[IDX_VEN] = venous_inflow - co * c_ven

        return dydt

    def simulate(
        self,
        t_end_h: float = 24.0,
        dt_h: float = 0.1,
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ) -> SimulationResult:
        """Run the PBPK simulation.

        Args:
            t_end_h: Simulation end time (h).
            dt_h: Output time step (h).
            rtol: Relative ODE solver tolerance.
            atol: Absolute ODE solver tolerance.

        Returns:
            SimulationResult with time, amounts, and computed PK parameters.
        """
        t_eval = np.arange(0.0, t_end_h + dt_h / 2, dt_h)

        sol = solve_ivp(
            self._rhs,
            t_span=(0.0, t_end_h),
            y0=self._y0,
            method="LSODA",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=0.1,
        )

        if not sol.success:
            logger.warning("ODE solver warning: %s", sol.message)

        amounts = sol.y.T.copy()

        # Post-solve check: warn if any state went significantly negative
        neg_min = np.min(amounts, axis=0)  # minimum value per state
        for state_idx in range(N_STATES):
            if neg_min[state_idx] < -1e-6:
                logger.warning(
                    "Negative state detected: state[%d] min=%.3e mg. "
                    "Consider tightening solver tolerances.",
                    state_idx,
                    float(neg_min[state_idx]),
                )

        return SimulationResult(
            time_h=sol.t,
            amounts=amounts,
            drug=self.drug,
            organs=self.organs,
        )
