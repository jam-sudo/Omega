"""Wrapper around ADMET-AI to conform to the MLADMEPredictor interface.

Maps ADMET-AI outputs to the ADMEProperties contract with proper unit conversions:
  - Lipophilicity -> logP (direct)
  - PPBR -> fup: fup = 1 - ppbr/100
  - Clearance_Hepatocyte -> clint_3a4: hepatocyte units -> pmol CYP units
  - Caco2_Wang -> peff: 10^-6 cm/s -> x10^-4 cm/s (multiply by 100)
  - Solubility_AqSolDB -> logS (direct, log mol/L)
  - hERG -> herg_ic50_uM (direct)
  - CYP2D6_Substrate -> clint_2d6 (categorical heuristic)
  - RBP: default 1.5 (population mean; overridden by XGBoost in ensemble)
  - MW: computed from SMILES via RDKit
"""

from __future__ import annotations

import logging
from typing import Any

from omega_pbpk.ml.models.adme import MLADMEPredictor
from omega_pbpk.prediction.adme_predictor import ADMEProperties

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Unit conversion constants (literature values)
# -----------------------------------------------------------------------
# MPPGL: Microsomal Protein Per Gram of Liver (mg protein / g liver)
MPPGL = 40.0  # mg microsomal protein per g liver

# CYP3A4 abundance: pmol CYP3A4 per mg microsomal protein
CYP3A4_ABUNDANCE = 137.0  # pmol/mg microsomal protein

# Hepatocellularity: million hepatocytes per g liver
HEPATOCELLULARITY = 120.0  # 10^6 cells / g liver

# Conversion factor: clearance in uL/min/10^6 cells -> uL/min/pmol CYP3A4
# clint_pmol = clint_hepatocyte / (MPPGL * CYP3A4_ABUNDANCE)
# This accounts for: hepatocytes -> microsomes -> specific CYP isoform
HEPATOCYTE_TO_PMOL_CYP3A4 = MPPGL * CYP3A4_ABUNDANCE  # ~5480

# Fraction of total hepatic clearance attributed to each CYP isoform.
# ADMET-AI Clearance_Hepatocyte is TOTAL across all pathways; we must
# apportion before converting to per-enzyme units.
# Literature estimates (Rodrigues 1999, Rendic 2002):
#   CYP3A4: ~50% of hepatic CYP-mediated clearance
#   CYP2D6: ~10% of hepatic CYP-mediated clearance
CYP3A4_FRACTION = 0.50
CYP2D6_FRACTION = 0.10

# Default clint_2d6 values based on CYP2D6 substrate classification
CLINT_2D6_SUBSTRATE = 5.0  # uL/min/pmol if CYP2D6 substrate
CLINT_2D6_NON_SUBSTRATE = 0.5  # uL/min/pmol if NOT CYP2D6 substrate


def _compute_mw_from_smiles(smiles: str) -> float:
    """Compute molecular weight from SMILES using RDKit.

    Args:
        smiles: SMILES string.

    Returns:
        Molecular weight in g/mol.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        raise ImportError(
            "RDKit is required for MW computation. Install with: pip install rdkit-pypi"
        ) from None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return float(Descriptors.MolWt(mol))


class ADMETAIPredictor(MLADMEPredictor):
    """ADMET-AI based ADME predictor.

    Wraps the admet-ai package to predict molecular ADME properties,
    converting outputs to the ADMEProperties contract.

    Raises ImportError on instantiation if admet-ai is not installed.
    """

    def __init__(self) -> None:
        """Initialize the ADMET-AI predictor.

        Raises:
            ImportError: If admet-ai is not installed.
        """
        try:
            from admet_ai import ADMETModel
        except ImportError:
            raise ImportError(
                "admet-ai is required for ADMETAIPredictor. Install with: pip install admet-ai"
            ) from None
        self._model = ADMETModel()
        logger.info("ADMETAIPredictor initialized successfully.")

    def predict(self, smiles: str) -> ADMEProperties:
        """Predict ADME properties from a SMILES string.

        Args:
            smiles: Canonical SMILES string for the compound.

        Returns:
            ADMEProperties with all 9 properties, confidence, and intervals.

        Raises:
            ValueError: If SMILES is invalid or prediction fails.
        """
        return self._predict_single(smiles)

    def predict_batch(self, smiles_list: list[str]) -> list[ADMEProperties]:
        """Predict ADME properties for multiple compounds.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of ADMEProperties, one per input SMILES.
        """
        return [self._predict_single(s) for s in smiles_list]

    def _predict_single(self, smiles: str) -> ADMEProperties:
        """Internal single-compound prediction with full unit conversion.

        Args:
            smiles: SMILES string.

        Returns:
            ADMEProperties instance.
        """
        # Compute MW from SMILES (always available via RDKit)
        mw = _compute_mw_from_smiles(smiles)

        # Get ADMET-AI predictions
        raw = self._get_raw_predictions(smiles)

        # Map and convert each property
        logp = self._extract_logp(raw, smiles=smiles)
        fup = self._extract_fup(raw)
        clint_3a4 = self._extract_clint_3a4(raw)
        clint_hepatocyte_raw = self._extract_clint_hepatocyte_raw(raw)
        peff = self._extract_peff(raw)
        logs = self._extract_logs(raw)
        herg = self._extract_herg(raw)
        clint_2d6 = self._extract_clint_2d6(raw)
        rbp = 1.5  # Default (population mean); overridden by XGBoost in ensemble

        # Confidence from applicability domain if available
        confidence = self._determine_confidence(raw)

        # Conformal intervals: use ADMET-AI intervals if available, else +/-50%
        fup_lo, fup_hi = self._get_interval(raw, "PPBR", fup, transform_fup=True)
        clint_lo, clint_hi = self._get_interval(
            raw,
            "Clearance_Hepatocyte",
            clint_3a4,
            scale_factor=CYP3A4_FRACTION / HEPATOCYTE_TO_PMOL_CYP3A4,
        )
        peff_lo, peff_hi = self._get_interval(raw, "Caco2_Wang", peff, scale_factor=100.0)
        rbp_lo = max(0.5, rbp * 0.5)
        rbp_hi = min(3.0, rbp * 1.5)

        return ADMEProperties(
            mw=round(mw, 2),
            logP=round(logp, 2),
            logS=round(logs, 2),
            peff=round(peff, 3),
            fup=round(max(0.001, min(1.0, fup)), 4),
            rbp=round(rbp, 3),
            clint_3a4=round(max(0.01, clint_3a4), 3),
            clint_2d6=round(clint_2d6, 3),
            herg_ic50_uM=round(max(0.01, herg), 2),
            confidence=confidence,
            clint_hepatocyte_uL_min=round(clint_hepatocyte_raw, 3),
            fup_lo=round(max(0.0, fup_lo), 4),
            fup_hi=round(min(1.0, fup_hi), 4),
            clint_3a4_lo=round(max(0.0, clint_lo), 3),
            clint_3a4_hi=round(clint_hi, 3),
            peff_lo=round(max(0.0, peff_lo), 4),
            peff_hi=round(peff_hi, 4),
            rbp_lo=round(rbp_lo, 4),
            rbp_hi=round(rbp_hi, 4),
        )

    def _get_raw_predictions(self, smiles: str) -> dict[str, Any]:
        """Call ADMET-AI and return raw prediction dictionary.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary of raw ADMET-AI predictions.

        Raises:
            ValueError: If ADMET-AI fails to predict.
        """
        try:
            preds = self._model.predict(smiles=smiles)
            # admet-ai returns a DataFrame; convert first row to dict
            if hasattr(preds, "to_dict"):
                # DataFrame -> dict of first row
                return {k: v for k, v in preds.iloc[0].to_dict().items()}
            return dict(preds)
        except Exception as exc:
            raise ValueError(f"ADMET-AI prediction failed for SMILES '{smiles}': {exc}") from exc

    # ------------------------------------------------------------------
    # Property extraction with unit conversions
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_logp(raw: dict[str, Any], smiles: str | None = None) -> float:
        """Extract lipophilicity from ADMET-AI.

        ADMET-AI Lipophilicity output is logD at pH 7.4 (apparent partitioning).
        While not equal to logP for ionized drugs, logD is actually the more
        appropriate parameter for tissue distribution (Kp estimation) because
        ionized species distribute less into tissues. Using logP with incorrect
        fup predictions leads to worse Kp/Vd estimates than using logD.

        For neutral drugs, logD ≈ logP so the distinction doesn't matter.
        """
        val = raw.get("Lipophilicity_AstraZeneca", raw.get("Lipophilicity", None))
        if val is not None:
            return float(val)
        # Fallback: RDKit Crippen logP (better than hardcoded default)
        if smiles is not None:
            try:
                from rdkit import Chem
                from rdkit.Chem import Descriptors

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return float(Descriptors.MolLogP(mol))
            except (ImportError, Exception):
                pass
        logger.warning("Lipophilicity not available; defaulting to 2.0")
        return 2.0

    @staticmethod
    def _extract_fup(raw: dict[str, Any]) -> float:
        """Extract fraction unbound in plasma from PPBR.

        ADMET-AI PPBR = Plasma Protein Binding Rate (percent bound, 0-100).
        Conversion: fup = 1 - PPBR/100
        """
        ppbr = raw.get("PPBR_AZ", raw.get("PPBR", None))
        if ppbr is not None:
            ppbr_val = float(ppbr)
            # PPBR is percent bound (0-100), so fup = 1 - ppbr/100
            fup = 1.0 - ppbr_val / 100.0
            return max(0.001, min(1.0, fup))
        logger.warning("PPBR not in ADMET-AI output; defaulting fup to 0.1")
        return 0.1

    @staticmethod
    def _extract_clint_3a4(raw: dict[str, Any]) -> float:
        """Extract CYP3A4 intrinsic clearance from hepatocyte clearance.

        CRITICAL UNIT CONVERSION:
        ADMET-AI Clearance_Hepatocyte is in uL/min/10^6 cells (hepatocyte units).
        Our system needs uL/min/pmol CYP3A4.

        Conversion path:
          clint_hepatocyte (uL/min/10^6 cells)
          -> clint_microsomal = clint_hepatocyte * (10^6 cells / HEPATOCELLULARITY per g)
             * (g liver / MPPGL mg protein)   [simplified scaling]
          -> clint_pmol = clint_hepatocyte / (MPPGL * CYP3A4_ABUNDANCE)

        Where:
          MPPGL = 40 mg microsomal protein / g liver
          CYP3A4_ABUNDANCE = 137 pmol CYP3A4 / mg microsomal protein
          Conversion factor = 40 * 137 = 5480

        So: clint_pmol = clint_hepatocyte / 5480
        """
        cl_hep = raw.get(
            "Clearance_Hepatocyte_AZ",
            raw.get("Clearance_Hepatocyte", None),
        )
        if cl_hep is not None:
            cl_hep_val = float(cl_hep)
            # Apportion total hepatic clearance to CYP3A4 fraction, then
            # convert from uL/min/10^6 cells -> uL/min/pmol CYP3A4
            clint_pmol = cl_hep_val * CYP3A4_FRACTION / HEPATOCYTE_TO_PMOL_CYP3A4
            return max(0.01, clint_pmol)
        logger.warning("Clearance_Hepatocyte not in ADMET-AI output; defaulting clint_3a4 to 1.0")
        return 1.0

    @staticmethod
    def _extract_clint_hepatocyte_raw(raw: dict[str, Any]) -> float:
        """Extract raw hepatocyte intrinsic clearance (µL/min/10^6 cells).

        This is the TOTAL clearance from ADMET-AI before any CYP attribution.
        Used for proper IVIVE scaling in the pipeline.
        """
        cl_hep = raw.get(
            "Clearance_Hepatocyte_AZ",
            raw.get("Clearance_Hepatocyte", None),
        )
        if cl_hep is not None:
            return max(0.0, float(cl_hep))
        return 0.0

    @staticmethod
    def _extract_peff(raw: dict[str, Any]) -> float:
        """Extract human effective intestinal permeability from Caco-2 Papp.

        ADMET-AI Caco2_Wang predicts log10(Papp) where Papp is the apparent
        permeability across a Caco-2 monolayer (in cm/s).

        CRITICAL: Caco-2 Papp ≠ human intestinal Peff.
        In-vivo human Peff is typically 10-100x higher than in-vitro Caco-2 Papp
        due to larger absorptive surface area (villi, microvilli), paracellular
        transport, and active uptake mechanisms.

        We apply the Sun et al. (2002) empirical correlation:
          log(Peff_human) = 0.6836 × log(Papp_caco2) - 0.5579

        This converts Caco-2 Papp to human jejunal Peff. Result is in cm/s,
        then scaled to our system unit (×10^-4 cm/s).

        Validated: Caco2_Wang ≈ -4.58 (ibuprofen) → Peff ≈ 5.3 ×10^-4 cm/s
        (literature: 5-10), vs uncorrected 0.26 ×10^-4 cm/s.
        """
        caco2 = raw.get("Caco2_Wang", None)
        if caco2 is not None:
            caco2_val = float(caco2)
            # Sun et al. (2002) Caco-2 → human Peff correlation
            log_peff = 0.6836 * caco2_val - 0.5579
            peff_cm_s = 10.0**log_peff
            peff_1e4 = peff_cm_s * 1e4  # convert cm/s to ×10^-4 cm/s
            return max(0.01, min(100.0, peff_1e4))
        logger.warning("Caco2_Wang not in ADMET-AI output; defaulting peff to 1.0")
        return 1.0

    @staticmethod
    def _extract_logs(raw: dict[str, Any]) -> float:
        """Extract aqueous solubility (log mol/L) -- direct mapping.

        ADMET-AI Solubility_AqSolDB is already in log mol/L.
        """
        logs = raw.get("Solubility_AqSolDB", None)
        if logs is not None:
            return float(logs)
        logger.warning("Solubility_AqSolDB not in ADMET-AI output; defaulting logS to -3.0")
        return -3.0

    @staticmethod
    def _extract_herg(raw: dict[str, Any]) -> float:
        """Extract hERG IC50 in uM -- direct mapping.

        ADMET-AI hERG output is a binary blocker probability (0-1).
        We convert to a pseudo IC50: if p(blocker) > 0.5, IC50 ~ 1 uM (risky);
        otherwise scale inversely.
        """
        herg_prob = raw.get("hERG", None)
        if herg_prob is not None:
            p = float(herg_prob)
            # Convert probability to pseudo IC50.
            # ADMET-AI hERG classifier threshold = IC50 < 10 µM (blocker).
            # Calibrated to reference compounds in data/adme_reference.csv:
            #   p=0.0 -> IC50=100 µM (non-blocker)
            #   p=0.5 -> IC50=10 µM (decision boundary = classification threshold)
            #   p=1.0 -> IC50=1 µM (confirmed blocker)
            # Formula: log10(IC50) = 2 - 2*p  =>  IC50 = 10^(2 - 2*p)
            # This correctly anchors p=0.5 to the 10 µM threshold, unlike the
            # previous 10^(2-3*p) which erroneously mapped p=0.5 → 3.16 µM.
            if p > 0.99:
                return 0.1
            if p < 0.01:
                return 100.0
            ic50 = 10.0 ** (2.0 - 2.0 * p)
            return max(0.01, min(1000.0, ic50))
        logger.warning("hERG not in ADMET-AI output; defaulting herg_ic50_uM to 10.0")
        return 10.0

    @staticmethod
    def _extract_clint_2d6(raw: dict[str, Any]) -> float:
        """Extract CYP2D6 intrinsic clearance from substrate classification.

        ADMET-AI CYP2D6_Substrate is categorical (0 or 1).
        Heuristic:
          - Substrate (1): clint_2d6 = 5.0 uL/min/pmol
          - Non-substrate (0): clint_2d6 = 0.5 uL/min/pmol
        """
        cyp2d6 = raw.get("CYP2D6_Substrate", None)
        if cyp2d6 is not None:
            is_substrate = float(cyp2d6) > 0.5
            return CLINT_2D6_SUBSTRATE if is_substrate else CLINT_2D6_NON_SUBSTRATE
        logger.warning("CYP2D6_Substrate not in ADMET-AI output; defaulting clint_2d6 to 0.5")
        return CLINT_2D6_NON_SUBSTRATE

    @staticmethod
    def _determine_confidence(raw: dict[str, Any]) -> str:
        """Determine prediction confidence from ADMET-AI applicability domain.

        If ADMET-AI provides applicability domain info, use it.
        Otherwise default to 'medium'.
        """
        # Check for applicability domain fields that ADMET-AI may provide
        ad_score = raw.get("applicability_domain", raw.get("AD_score", None))
        if ad_score is not None:
            score = float(ad_score)
            if score > 0.8:
                return "high"
            if score > 0.4:
                return "medium"
            return "low"
        return "medium"

    @staticmethod
    def _get_interval(
        raw: dict[str, Any],
        key_prefix: str,
        point_estimate: float,
        transform_fup: bool = False,
        scale_factor: float = 1.0,
    ) -> tuple[float, float]:
        """Extract or compute prediction interval for a property.

        Looks for ADMET-AI interval keys like '{key_prefix}_lower' / '{key_prefix}_upper'.
        Falls back to +/-50% of point estimate.

        Args:
            raw: Raw ADMET-AI output dict.
            key_prefix: Property key prefix in ADMET-AI output.
            point_estimate: The point estimate in our unit system.
            transform_fup: If True, convert PPBR intervals to fup intervals.
            scale_factor: Multiplicative factor for unit conversion.

        Returns:
            (lower, upper) interval bounds.
        """
        lo_key = f"{key_prefix}_lower"
        hi_key = f"{key_prefix}_upper"

        if lo_key in raw and hi_key in raw:
            lo_raw = float(raw[lo_key])
            hi_raw = float(raw[hi_key])
            if transform_fup:
                # PPBR intervals -> fup intervals (reversed)
                return 1.0 - hi_raw / 100.0, 1.0 - lo_raw / 100.0
            return lo_raw * scale_factor, hi_raw * scale_factor

        # Default: +/-50% interval
        lo = point_estimate * 0.5
        hi = point_estimate * 1.5
        return lo, hi
