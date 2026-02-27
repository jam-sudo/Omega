"""Full in vitro → in vivo extrapolation (IVIVE) pipeline.

Automates the complete translation from a SMILES string to a fully
parameterised Drug object that can be fed into WholeBodyPBPK, using
in vitro assay simulations as an intermediate calibration step.

Pipeline stages
---------------
1. **ADME prediction** (ADMEPredictor)
   SMILES → MW, logP, fup, rbp, Peff, CLint_3A4, CLint_2D6, hERG IC50

2. **Phase 2 prediction** (Phase2Predictor)
   SMILES → CLint_UGT, CLint_SULT, CLint_MAO

3. **Assay simulation** (MicrosomesAssay or HepatocyteAssay)
   Simulated depletion curve → k_dep → whole-liver CLint

4. **Caco-2 simulation** (Caco2Assay, optional)
   Papp_AB, Papp_BA, efflux ratio, P-gp/BCRP flag

5. **Receptor occupancy** (EquilibriumOccupancy, optional)
   Kd → occupancy-time curve → EC50 (for PD linking)

6. **IVIVE scaling** (clinical.ivive)
   CLint_total (Phase 1 + Phase 2) → CLh (well-stirred model)

7. **Drug construction**
   All parameters → Drug dataclass → ready for PBPK simulation

Usage
-----
    from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest

    req = InVitroRequest(
        smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",   # caffeine
        dose_mg=100.0,
        assay_type="hepatocyte",
        run_caco2=True,
        kd_mg_L=0.05,                            # optional receptor Kd
    )
    pipeline = InVitroPipeline()
    result = pipeline.run(req)

    # Access individual stage results
    print(result.clint_total_ul_min_mg)           # total in vitro CLint
    print(result.clh_l_per_h)                     # predicted in vivo CLh
    print(result.caco2.efflux_ratio)              # transporter flag
    print(result.drug)                             # Drug object for PBPK
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from omega_pbpk.in_vitro.assay import AssayResult, HepatocyteAssay, MicrosomesAssay
from omega_pbpk.in_vitro.caco2 import Caco2Assay, Caco2Result
from omega_pbpk.in_vitro.receptor import (
    EC50Result,
    EC50Extractor,
    EquilibriumOccupancy,
    OperationalModel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InVitroRequest:
    """Input specification for the in vitro pipeline.

    Attributes:
        smiles: SMILES string of the drug molecule.
        dose_mg: Dose for the downstream PBPK simulation (mg).
        assay_type: "microsomes" (Phase 1 only) or "hepatocyte"
            (Phase 1 + Phase 2 combined).
        c0_uM: Initial drug concentration in the assay well (µM).
        run_caco2: If True, simulate Caco-2 bidirectional permeability.
        kd_mg_L: Receptor Kd (mg/L) for occupancy modelling.
            Set to None to skip receptor analysis.
        tau: Operational model transduction ratio (dimensionless).
            Only used if kd_mg_L is provided.
        protein_conc_mg_mL: Microsomal protein concentration (mg/mL).
            Used for MicrosomesAssay.
        cell_density_million_per_mL: Hepatocyte density (10⁶/mL).
            Used for HepatocyteAssay.
        fu_mic: Microsomal binding correction (fu_mic).
            If None, estimated from logP via Austin (2002) equation.
        drug_name: Display name for the generated Drug object.
    """

    smiles: str
    dose_mg: float = 100.0
    assay_type: str = "hepatocyte"          # "microsomes" or "hepatocyte"
    c0_uM: float = 1.0
    run_caco2: bool = True
    kd_mg_L: float | None = None
    tau: float = 1.0
    protein_conc_mg_mL: float = 0.5
    cell_density_million_per_mL: float = 1.0
    fu_mic: float | None = None
    drug_name: str = "predicted_compound"


@dataclass
class InVitroResult:
    """Complete output from the in vitro → in vivo pipeline.

    Attributes:
        request: Original input request.
        adme: ADMEProperties from ADMEPredictor.
        phase2: Phase2Properties from Phase2Predictor.
        assay: AssayResult from microsomal or hepatocyte assay.
        caco2: Caco2Result (or None if run_caco2=False).
        receptor: EC50Result (or None if kd_mg_L not provided).
        clint_phase1_ul_min_mg: Phase 1 CLint (CYP3A4 + CYP2D6, µL/min/mg).
        clint_phase2_ul_min_mg: Phase 2 CLint (UGT + SULT + MAO, µL/min/mg).
        clint_total_ul_min_mg: Total CLint (Phase 1 + Phase 2).
        clh_l_per_h: Predicted in vivo hepatic clearance (well-stirred, L/h).
        drug: Constructed Drug object ready for PBPK simulation.
        warnings: Aggregated warnings from all pipeline stages.
        confidence: Overall confidence ("high", "medium", "low").
    """

    request: InVitroRequest
    adme: Any                           # ADMEProperties
    phase2: Any                         # Phase2Properties
    assay: AssayResult
    caco2: Caco2Result | None
    receptor: EC50Result | None
    clint_phase1_ul_min_mg: float
    clint_phase2_ul_min_mg: float
    clint_total_ul_min_mg: float
    clh_l_per_h: float
    drug: Any                           # Drug dataclass
    warnings: list[str] = field(default_factory=list)
    confidence: str = "low"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class InVitroPipeline:
    """Full in vitro → in vivo extrapolation pipeline.

    Thread-safe; ADME predictor and Phase 2 predictor are stateless.
    """

    def __init__(self) -> None:
        self._adme: Any = None
        self._phase2: Any = None

    def _ensure_predictors(self) -> None:
        if self._adme is None:
            from omega_pbpk.prediction.adme_predictor import ADMEPredictor
            self._adme = ADMEPredictor()
        if self._phase2 is None:
            from omega_pbpk.prediction.phase2_predictor import Phase2Predictor
            self._phase2 = Phase2Predictor()

    def run(self, request: InVitroRequest) -> InVitroResult:
        """Execute all pipeline stages.

        Args:
            request: InVitroRequest with SMILES and assay configuration.

        Returns:
            InVitroResult with all stage outputs and the final Drug object.
        """
        self._ensure_predictors()
        warnings: list[str] = []

        # ------------------------------------------------------------------
        # Stage 1: ADME prediction
        # ------------------------------------------------------------------
        adme = self._adme.predict(request.smiles)
        logger.info("ADME prediction complete (confidence=%s)", adme.confidence)

        # ------------------------------------------------------------------
        # Stage 2: Phase 2 prediction
        # ------------------------------------------------------------------
        phase2 = self._phase2.predict(request.smiles)
        logger.info(
            "Phase 2 prediction: UGT=%.3f, SULT=%.3f, MAO=%.3f µL/min/mg",
            phase2.clint_ugt_ul_min_mg,
            phase2.clint_sult_ul_min_mg,
            phase2.clint_mao_ul_min_mg,
        )

        # ------------------------------------------------------------------
        # Stage 3: Assay simulation
        # ------------------------------------------------------------------
        clint_phase1 = adme.clint_3a4 + adme.clint_2d6
        clint_phase2 = phase2.clint_phase2_total_ul_min_mg

        # fu_mic: estimate from logP if not provided
        fu_mic = request.fu_mic
        if fu_mic is None:
            fu_mic = MicrosomesAssay.estimate_fu_mic(adme.logP)
            logger.debug("Estimated fu_mic = %.3f from logP = %.2f", fu_mic, adme.logP)

        if request.assay_type == "microsomes":
            assay_sim = MicrosomesAssay(
                protein_conc_mg_mL=request.protein_conc_mg_mL,
                f_unbound_mic=fu_mic,
            )
            assay_result = assay_sim.simulate(
                clint_ul_min_mg=clint_phase1,
                c0_uM=request.c0_uM,
            )
        else:
            # Hepatocyte: total Phase 1 + Phase 2 CLint (combined in cells)
            clint_total_for_hep = _convert_phase1_to_hepatocyte_units(clint_phase1) + \
                                  _convert_phase2_to_hepatocyte_units(clint_phase2)
            assay_sim_hep = HepatocyteAssay(
                cell_density_million_per_mL=request.cell_density_million_per_mL,
                fu_cell=fu_mic,  # use fu_mic as proxy for fu_cell
            )
            assay_result = assay_sim_hep.simulate(
                clint_ul_min_million=clint_total_for_hep,
                c0_uM=request.c0_uM,
            )

        warnings.extend(assay_result.warnings)

        # ------------------------------------------------------------------
        # Stage 4: Caco-2 permeability (optional)
        # ------------------------------------------------------------------
        caco2_result: Caco2Result | None = None
        if request.run_caco2:
            # Estimate Papp and efflux from adme logP (and TPSA if RDKit available)
            papp_passive = _estimate_papp(adme.logP, adme.peff)
            # Efflux: elevated for high logP + high MW (P-gp substrate risk)
            efflux_multiplier = _estimate_efflux_multiplier(adme.logP, adme.mw)
            papp_ba = papp_passive * efflux_multiplier

            caco2_sim = Caco2Assay(
                papp_ab_cm_s=papp_passive,
                papp_ba_cm_s=papp_ba,
            )
            caco2_result = caco2_sim.simulate(c0_apical_uM=request.c0_uM)
            warnings.extend(caco2_result.warnings)
            logger.info(
                "Caco-2: Papp(A→B)=%.2f, ER=%.2f → %s",
                caco2_result.papp_ab_cm_s,
                caco2_result.efflux_ratio,
                caco2_result.pgp_bcrp_flag,
            )

        # ------------------------------------------------------------------
        # Stage 5: Receptor occupancy / EC50 (optional)
        # ------------------------------------------------------------------
        receptor_result: EC50Result | None = None
        if request.kd_mg_L is not None:
            occ_model = EquilibriumOccupancy(kd_mg_L=request.kd_mg_L)
            op_model = OperationalModel(tau=request.tau, emax=100.0, e0=0.0)
            extractor = EC50Extractor(
                occupancy_model=occ_model,
                operational_model=op_model,
            )
            # Use a representative Cmax range (0.1–100× Kd)
            import numpy as np
            kd = request.kd_mg_L
            cmax_range = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]) * kd
            doses_est = cmax_range * 50.0  # rough dose proxy
            receptor_result = extractor.fit_from_cmax_array(cmax_range, doses_est)
            logger.info(
                "Receptor EC50 extracted: %.4f mg/L (R²=%.3f)",
                receptor_result.ec50_mg_L,
                receptor_result.r_squared,
            )

        # ------------------------------------------------------------------
        # Stage 6: IVIVE scaling
        # ------------------------------------------------------------------
        from omega_pbpk.clinical.ivive import scale_microsomal_clint

        clint_total = clint_phase1 + clint_phase2
        ivive_result = scale_microsomal_clint(
            clint_uL_min_mg=clint_total,
            fup=adme.fup,
            fu_mic=fu_mic,
        )
        clh = ivive_result.clh_well_stirred_L_per_h
        logger.info(
            "IVIVE: CLint_total=%.3f µL/min/mg → CLh=%.3f L/h",
            clint_total, clh,
        )

        # ------------------------------------------------------------------
        # Stage 7: Drug construction
        # ------------------------------------------------------------------
        drug = _build_drug(request, adme, clh, caco2_result)

        # Overall confidence: minimum across ADME and Phase 2
        conf_rank = {"high": 2, "medium": 1, "low": 0}
        min_conf = min(
            conf_rank.get(adme.confidence, 0),
            conf_rank.get(phase2.confidence, 0),
        )
        confidence = ["low", "medium", "high"][min_conf]

        if caco2_result and caco2_result.efflux_ratio > 2.0:
            warnings.append(
                f"P-gp/BCRP efflux substrate detected (ER={caco2_result.efflux_ratio:.1f}); "
                "oral bioavailability may be lower than PBPK predicts."
            )
        if adme.herg_ic50_uM < 1.0:
            warnings.append(
                f"hERG IC50 = {adme.herg_ic50_uM:.2f} µM — cardiac safety concern; "
                "safety margin should be checked against predicted Cmax."
            )

        return InVitroResult(
            request=request,
            adme=adme,
            phase2=phase2,
            assay=assay_result,
            caco2=caco2_result,
            receptor=receptor_result,
            clint_phase1_ul_min_mg=round(clint_phase1, 4),
            clint_phase2_ul_min_mg=round(clint_phase2, 4),
            clint_total_ul_min_mg=round(clint_total, 4),
            clh_l_per_h=round(clh, 4),
            drug=drug,
            warnings=warnings,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _convert_phase1_to_hepatocyte_units(clint_ul_min_mg: float) -> float:
    """Convert microsomal CLint (µL/min/mg) to hepatocyte units (µL/min/10⁶ cells).

    Using MPPGL = 40 mg/g and HPGL = 120×10⁶ cells/g:
        CLint_hep = CLint_mic × MPPGL / (HPGL/1e6)
                  = CLint_mic × 40 / 120
    """
    return clint_ul_min_mg * 40.0 / 120.0


def _convert_phase2_to_hepatocyte_units(clint_ul_min_mg: float) -> float:
    """Convert Phase 2 CLint (µL/min/mg) to hepatocyte units.

    Phase 2 enzymes (UGTs) are enriched in hepatocytes relative to
    microsomes; scaling assumes 50% of UGT activity is in microsomes:
        CLint_hep_phase2 ≈ CLint_mic_phase2 × 2 × MPPGL / (HPGL/1e6)
    """
    return clint_ul_min_mg * 2.0 * 40.0 / 120.0


def _estimate_papp(logP: float, peff_cm_s: float) -> float:
    """Estimate Caco-2 Papp from intestinal Peff (m/s → cm/s conversion).

    Papp (cm/s) ≈ Peff × 0.3  (empirical factor, Sun 2002)
    """
    return float(max(peff_cm_s * 1e-4 * 0.3, 1e-9))  # Peff in ×10⁻⁴ cm/s


def _estimate_efflux_multiplier(logP: float, mw: float) -> float:
    """Estimate B→A efflux multiplier relative to passive Papp.

    P-gp substrate risk increases with high logP (> 4) and high MW (> 400).
    Based on Polli et al. J Med Chem. 2001;44(12):1946-53.
    """
    risk = 0.0
    if logP > 4.0:
        risk += (logP - 4.0) * 0.3
    if mw > 400.0:
        risk += (mw - 400.0) / 400.0 * 0.5
    return float(1.0 + min(risk, 8.0))  # cap at 9× passive


def _build_drug(
    request: InVitroRequest,
    adme: Any,
    clh_l_per_h: float,
    caco2_result: Caco2Result | None,
) -> Any:
    """Construct a Drug dataclass from pipeline outputs."""
    from omega_pbpk.drugs.drug import Drug

    # Effective Peff: if Caco-2 result available, use simulated Papp
    peff = adme.peff
    if caco2_result is not None:
        # Convert Papp (×10⁻⁶ cm/s) back to Peff (×10⁻⁴ cm/s) using reverse scaling
        peff = max(caco2_result.papp_ab_cm_s / 0.3 * 1e2, 0.01)

    return Drug(
        name=request.drug_name,
        mw=adme.mw,
        logP=adme.logP,
        fup=max(adme.fup, 0.001),
        rbp=adme.rbp,
        peff=peff,
        clint_hepatic_L_per_h=clh_l_per_h,
        clint={"CYP3A4": adme.clint_3a4, "CYP2D6": adme.clint_2d6},
    )


__all__ = ["InVitroRequest", "InVitroResult", "InVitroPipeline"]
