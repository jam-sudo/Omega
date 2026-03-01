"""Transporter substrate/inhibitor classification using logistic QSPR models.

Predicts 7 drug transporter interactions (substrate and inhibitor probability)
from physicochemical properties (MW, logP, charge class):

  - OATP1B1 (hepatic uptake)
  - OATP1B3 (hepatic uptake)
  - P-gp / P_gp (efflux, intestine/BBB)
  - BCRP (efflux, intestine/BBB)
  - OCT2 (renal uptake)
  - MRP2 (efflux, intestine/kidney)
  - MATE1 (renal efflux)

Reference: FDA DDI Guidance 2020; ITC White Paper (Giacomini et al., 2010).
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

TRANSPORTER_NAMES: list[str] = [
    "OATP1B1",
    "OATP1B3",
    "P_gp",
    "BCRP",
    "OCT2",
    "MRP2",
    "MATE1",
]

# CSV column mapping: transporter -> (substrate_col, inhibitor_col)
_CSV_COLS: dict[str, tuple[str, str]] = {
    "OATP1B1": ("oatp1b1_substrate", "oatp1b1_inhibitor"),
    "OATP1B3": ("oatp1b3_substrate", "oatp1b3_inhibitor"),
    "P_gp": ("pgp_substrate", "pgp_inhibitor"),
    "BCRP": ("bcrp_substrate", "bcrp_inhibitor"),
    "OCT2": ("oct2_substrate", "oct2_inhibitor"),
    "MRP2": ("mrp2_substrate", "mrp2_inhibitor"),
    "MATE1": ("mate1_substrate", "mate1_inhibitor"),
}

_BASE_KM: dict[str, float] = {
    "OATP1B1": 10.0,
    "OATP1B3": 15.0,
    "P_gp": 50.0,
    "BCRP": 20.0,
    "OCT2": 100.0,
    "MRP2": 30.0,
    "MATE1": 80.0,
}

_BASE_IC50: dict[str, float] = {
    "OATP1B1": 5.0,
    "OATP1B3": 8.0,
    "P_gp": 20.0,
    "BCRP": 10.0,
    "OCT2": 50.0,
    "MRP2": 15.0,
    "MATE1": 40.0,
}


@dataclass(frozen=True)
class TransporterInteraction:
    """Substrate/inhibitor probabilities and kinetic parameters for one transporter."""

    transporter: str
    substrate_prob: float  # [0, 1]
    inhibitor_prob: float  # [0, 1]
    km_uM: float  # Km if substrate, 0.0 if not a substrate
    ic50_uM: float  # IC50 if inhibitor, float('inf') if not an inhibitor


@dataclass(frozen=True)
class TransporterProfile:
    """Predicted transporter interaction profile for a compound."""

    profiles: dict  # str -> TransporterInteraction
    confidence: str = "low"
    ddi_risk_flags: tuple = ()  # human-readable risk warnings

    def is_substrate(self, transporter: str, threshold: float = 0.5) -> bool:
        """Return True if substrate probability exceeds threshold."""
        interaction = self.profiles.get(transporter)
        if interaction is None:
            return False
        return interaction.substrate_prob >= threshold

    def is_inhibitor(self, transporter: str, threshold: float = 0.5) -> bool:
        """Return True if inhibitor probability exceeds threshold."""
        interaction = self.profiles.get(transporter)
        if interaction is None:
            return False
        return interaction.inhibitor_prob >= threshold

    def active_substrates(self, threshold: float = 0.5) -> list[str]:
        """Return list of transporters for which compound is a predicted substrate."""
        return [t for t, p in self.profiles.items() if p.substrate_prob >= threshold]

    def active_inhibitors(self, threshold: float = 0.5) -> list[str]:
        """Return list of transporters for which compound is a predicted inhibitor."""
        return [t for t, p in self.profiles.items() if p.inhibitor_prob >= threshold]


class TransporterClassifier:
    """QSPR-based transporter substrate/inhibitor classifier.

    Uses logistic regression models (binary cross-entropy + L2) fitted on
    data/transporter_reference.csv when sufficient positive examples exist.
    Falls back to rule-based heuristics for sparse transporters.

    Nearest-neighbour confidence scoring uses a 2-feature [MW/500, logP/5]
    Euclidean distance to the reference dataset.
    """

    _ref_data: list[dict] | None = None
    _ref_features: np.ndarray | None = None  # shape [N, 2] for NN distance

    # Fitted logistic regression weights per transporter/role
    _substrate_coeffs: dict[str, np.ndarray | None] = field(default_factory=dict)
    _inhibitor_coeffs: dict[str, np.ndarray | None] = field(default_factory=dict)

    def __init__(self) -> None:
        self._substrate_coeffs: dict[str, np.ndarray | None] = {}
        self._inhibitor_coeffs: dict[str, np.ndarray | None] = {}

    # ------------------------------------------------------------------
    # Reference data loading & model fitting
    # ------------------------------------------------------------------

    def _load_reference_data(self) -> None:
        """Load transporter_reference.csv and fit logistic regression models."""
        ref_path = (
            Path(__file__).parent.parent.parent.parent  # src/ -> Omega/
            / "data"
            / "transporter_reference.csv"
        )
        if not ref_path.exists():
            logger.warning(
                "transporter_reference.csv not found at %s; using rule-based fallback.",
                ref_path,
            )
            self._ref_data = []
            self._ref_features = np.empty((0, 2))
            return

        rows: list[dict] = []
        with open(ref_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        self._ref_data = rows
        logger.debug("Loaded %d transporter reference compounds.", len(rows))

        if rows:
            mw_arr = np.array([float(r["mw"]) for r in rows])
            lp_arr = np.array([float(r["logP"]) for r in rows])
            self._ref_features = np.column_stack([mw_arr / 500.0, lp_arr / 5.0])
        else:
            self._ref_features = np.empty((0, 2))

        self._fit_models()

    def _fit_models(self) -> None:
        """Fit one logistic regression per transporter × {substrate, inhibitor}."""
        rows = self._ref_data
        if not rows:
            return

        try:
            mw_arr = np.array([float(r["mw"]) for r in rows])
            lp_arr = np.array([float(r["logP"]) for r in rows])
            cc_arr = np.array([r["charge_class"] for r in rows])
        except (KeyError, ValueError) as exc:
            logger.warning("Transporter model fitting skipped: %s", exc)
            return

        X = np.column_stack(
            [
                self._build_features(float(mw), float(lp), cc)
                for mw, lp, cc in zip(mw_arr, lp_arr, cc_arr, strict=True)
            ]
        ).T  # shape [N, 8]

        for transporter, (sub_col, inh_col) in _CSV_COLS.items():
            try:
                y_sub = np.array([float(r[sub_col]) for r in rows])
                y_inh = np.array([float(r[inh_col]) for r in rows])
            except (KeyError, ValueError) as exc:
                logger.debug("Skipping %s: %s", transporter, exc)
                self._substrate_coeffs[transporter] = None
                self._inhibitor_coeffs[transporter] = None
                continue

            n_pos_sub = int(y_sub.sum())
            n_pos_inh = int(y_inh.sum())

            self._substrate_coeffs[transporter] = (
                self._fit_logistic(X, y_sub) if n_pos_sub >= 3 else None
            )
            self._inhibitor_coeffs[transporter] = (
                self._fit_logistic(X, y_inh) if n_pos_inh >= 3 else None
            )

            logger.debug(
                "%s: sub pos=%d (fitted=%s), inh pos=%d (fitted=%s)",
                transporter,
                n_pos_sub,
                self._substrate_coeffs[transporter] is not None,
                n_pos_inh,
                self._inhibitor_coeffs[transporter] is not None,
            )

    @staticmethod
    def _fit_logistic(X: np.ndarray, y: np.ndarray, lam: float = 0.1) -> np.ndarray:
        """Fit class-weighted L2-regularised logistic regression via scipy minimize.

        Uses balanced class weights (neg/pos ratio) to handle the severe class
        imbalance typical in transporter datasets (~3-15% positive rate).

        Args:
            X: Feature matrix [N, 8].
            y: Binary labels [N].
            lam: L2 regularisation strength.

        Returns:
            Weight vector [8].
        """
        n_features = X.shape[1]
        w0 = np.zeros(n_features)

        # Balanced class weights: up-weight positives by neg/pos ratio
        n_pos = max(float(y.sum()), 1.0)
        n_neg = max(float(len(y) - n_pos), 1.0)
        pos_weight = n_neg / n_pos
        sample_weights = np.where(y == 1, pos_weight, 1.0)
        # Normalise so weights sum to N (keeps loss scale comparable)
        sample_weights = sample_weights * len(y) / sample_weights.sum()

        def loss_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
            logits = X @ w
            # Numerically stable sigmoid
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
            probs = np.clip(probs, 1e-9, 1.0 - 1e-9)
            bce = -np.mean(sample_weights * (y * np.log(probs) + (1 - y) * np.log(1 - probs)))
            reg = 0.5 * lam * np.dot(w, w)
            loss = bce + reg
            grad = X.T @ (sample_weights * (probs - y)) / len(y) + lam * w
            return float(loss), grad

        result = minimize(
            loss_and_grad,
            w0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 200, "ftol": 1e-9},
        )
        return result.x

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _build_features(mw: float, logp: float, charge_class: str) -> np.ndarray:
        """Return 8-feature vector for logistic regression.

        Features: [1, MW/500, logP/5, (MW/500)², (logP/5)², (MW/500)*(logP/5),
                   is_cation, is_anion]
        """
        mw_n = mw / 500.0
        lp_n = logp / 5.0
        is_cation = 1.0 if charge_class in ("base", "cation") else 0.0
        is_anion = 1.0 if charge_class in ("acid", "anion") else 0.0
        return np.array([1.0, mw_n, lp_n, mw_n**2, lp_n**2, mw_n * lp_n, is_cation, is_anion])

    # ------------------------------------------------------------------
    # Rule-based heuristics (fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_prob(
        transporter: str,
        role: str,
        mw: float,
        logp: float,
        charge_class: str,
    ) -> float:
        """Return soft-threshold probability estimate from structural rules.

        Used when logistic model cannot be fitted (< 3 positives).
        """
        is_anion = charge_class in ("acid", "anion")
        is_cation = charge_class in ("base", "cation")
        is_neutral = charge_class in ("neutral", "zwitterion")

        if transporter == "OATP1B1":
            if role == "substrate":
                if is_anion and 350 < mw < 900 and 0 < logp < 6:
                    return 0.7
                return 0.1
            else:  # inhibitor
                if mw > 400 and logp > 1.5:
                    return 0.5
                return 0.1

        elif transporter == "OATP1B3":
            if role == "substrate":
                if is_anion and 300 < mw < 1000 and -0.5 < logp < 7:
                    return 0.7
                return 0.1
            else:
                if mw > 400 and logp > 1.5:
                    return 0.5
                return 0.1

        elif transporter == "P_gp":
            if role == "substrate":
                prob = 0.6 if (mw > 400 and logp > 2) else 0.1
                if is_cation:
                    prob = float(np.clip(prob + 0.1, 0.0, 1.0))
                return prob
            else:
                if mw > 350 and logp > 2.5:
                    return 0.5
                return 0.1

        elif transporter == "BCRP":
            if role == "substrate":
                if (is_anion or is_neutral) and 200 < mw < 700 and -1 < logp < 5:
                    return 0.5
                return 0.1
            else:
                if is_anion and logp > 0:
                    return 0.4
                return 0.1

        elif transporter == "OCT2":
            if role == "substrate":
                if is_cation and mw < 600 and logp < 3:
                    return 0.7
                return 0.05
            else:
                if is_cation and mw < 700:
                    return 0.5
                return 0.05

        elif transporter == "MRP2":
            if role == "substrate":
                if is_anion and mw > 200:
                    return 0.4
                return 0.05
            else:
                if mw > 350:
                    return 0.3
                return 0.1

        elif transporter == "MATE1":
            if role == "substrate":
                if is_cation and mw < 500 and logp < 2:
                    return 0.7
                return 0.05
            else:
                if is_cation:
                    return 0.4
                return 0.05

        return 0.1

    # ------------------------------------------------------------------
    # Km / IC50 estimation
    # ------------------------------------------------------------------

    def _estimate_km(self, transporter: str, mw: float, logp: float) -> float:
        """Estimate Km (µM) from base value scaled by MW."""
        km = _BASE_KM.get(transporter, 50.0)
        return float(np.clip(km * (mw / 400.0) ** 0.5, 1.0, 1000.0))

    def _estimate_ic50(self, transporter: str, mw: float, logp: float) -> float:
        """Estimate IC50 (µM) from base value inversely scaled by logP."""
        ic50 = _BASE_IC50.get(transporter, 20.0)
        return float(np.clip(ic50 / max(logp, 0.5) ** 0.3, 0.1, 500.0))

    # ------------------------------------------------------------------
    # DDI risk flag generation
    # ------------------------------------------------------------------

    @staticmethod
    def _assess_ddi_risk(
        profiles: dict[str, TransporterInteraction],
    ) -> tuple[str, ...]:
        """Generate DDI risk warnings from transporter interaction profiles."""
        flags: list[str] = []

        pgp = profiles.get("P_gp")
        if pgp and pgp.substrate_prob > 0.6:
            flags.append(
                "P-gp substrate: narrow therapeutic index drugs (digoxin) may have altered exposure"
            )

        oatp1b1 = profiles.get("OATP1B1")
        oatp1b3 = profiles.get("OATP1B3")
        if (oatp1b1 and oatp1b1.inhibitor_prob > 0.6) or (oatp1b3 and oatp1b3.inhibitor_prob > 0.6):
            flags.append(
                "OATP1B1/1B3 inhibitor: statin exposure may be significantly increased (AUCR risk)"
            )

        oct2 = profiles.get("OCT2")
        mate1 = profiles.get("MATE1")
        if (oct2 and oct2.substrate_prob > 0.6) or (mate1 and mate1.substrate_prob > 0.6):
            flags.append(
                "OCT2/MATE1 substrate: renal elimination may be inhibited by "
                "trimethoprim/cimetidine"
            )

        return tuple(flags)

    # ------------------------------------------------------------------
    # Confidence scoring (nearest-neighbour)
    # ------------------------------------------------------------------

    def _nn_confidence(self, mw: float, logp: float) -> str:
        """Assign confidence based on distance to nearest reference compound."""
        if self._ref_features is None or len(self._ref_features) == 0:
            return "low"
        q = np.array([mw / 500.0, logp / 5.0])
        dists = np.linalg.norm(self._ref_features - q, axis=1)
        dist = float(dists.min())
        if dist < 0.5:
            return "high"
        elif dist < 1.5:
            return "medium"
        return "low"

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------

    def _predict_from_features(
        self,
        mw: float,
        logp: float,
        charge_class: str,
        smiles: str | None = None,
    ) -> TransporterProfile:
        """Predict transporter profile from physicochemical properties."""
        feat = self._build_features(mw, logp, charge_class)
        profiles: dict[str, TransporterInteraction] = {}

        for transporter in TRANSPORTER_NAMES:
            # Substrate probability
            sub_coeffs = self._substrate_coeffs.get(transporter)
            if sub_coeffs is not None:
                logit_sub = float(feat @ sub_coeffs)
                sub_prob = float(1.0 / (1.0 + np.exp(-np.clip(logit_sub, -500, 500))))
            else:
                sub_prob = self._rule_based_prob(transporter, "substrate", mw, logp, charge_class)

            # Inhibitor probability
            inh_coeffs = self._inhibitor_coeffs.get(transporter)
            if inh_coeffs is not None:
                logit_inh = float(feat @ inh_coeffs)
                inh_prob = float(1.0 / (1.0 + np.exp(-np.clip(logit_inh, -500, 500))))
            else:
                inh_prob = self._rule_based_prob(transporter, "inhibitor", mw, logp, charge_class)

            # Kinetic parameters
            km = self._estimate_km(transporter, mw, logp) if sub_prob >= 0.5 else 0.0
            ic50 = self._estimate_ic50(transporter, mw, logp) if inh_prob >= 0.5 else float("inf")

            profiles[transporter] = TransporterInteraction(
                transporter=transporter,
                substrate_prob=round(sub_prob, 4),
                inhibitor_prob=round(inh_prob, 4),
                km_uM=round(km, 2),
                ic50_uM=round(ic50, 2) if ic50 != float("inf") else float("inf"),
            )

        confidence = self._nn_confidence(mw, logp)
        ddi_flags = self._assess_ddi_risk(profiles)

        return TransporterProfile(
            profiles=profiles,
            confidence=confidence,
            ddi_risk_flags=ddi_flags,
        )

    # ------------------------------------------------------------------
    # SMILES-based helpers (no RDKit required)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_mw(smiles: str) -> float:
        """Rough MW estimate from SMILES atom composition."""
        atom_weights: dict[str, float] = {
            "Cl": 35.5,
            "Br": 80.0,
            "C": 12.0,
            "N": 14.0,
            "O": 16.0,
            "S": 32.0,
            "F": 19.0,
            "I": 127.0,
            "P": 31.0,
        }
        mw = 0.0
        i = 0
        while i < len(smiles):
            if i + 1 < len(smiles) and smiles[i : i + 2] in ("Cl", "Br"):
                mw += atom_weights[smiles[i : i + 2]]
                i += 2
            elif smiles[i] in atom_weights:
                mw += atom_weights[smiles[i]]
                i += 1
            elif smiles[i].lower() in ("c", "n", "o", "s"):
                mw += {"c": 12.0, "n": 14.0, "o": 16.0, "s": 32.0}[smiles[i].lower()]
                i += 1
            else:
                i += 1
        return max(mw * 1.1, 1.0)

    @staticmethod
    def _estimate_logp(smiles: str) -> float:
        """Rough logP estimate from SMILES."""
        n_polar = smiles.count("O") + smiles.count("N") + smiles.count("S")
        n_halogen = smiles.count("F") + smiles.count("Cl") + smiles.count("Br")
        heavy = sum(1 for c in smiles if c.isalpha() and c not in "()")
        return 0.1 * heavy + 0.3 * n_halogen - 0.4 * n_polar

    @staticmethod
    def _estimate_charge_class(smiles: str) -> str:
        """Estimate charge class from SMILES pattern matching.

        Detects acidic groups (carboxylic/sulfonic acids) and basic
        nitrogen atoms not part of amide bonds.
        """
        has_acid = (
            "C(=O)O" in smiles
            or "C(=O)[O-]" in smiles
            or "S(=O)(=O)O" in smiles
            or "P(=O)(O)O" in smiles
        )
        # Count all N atoms not part of amide (NC(=O) or C(=O)N) or aromatic ring
        all_n = len(re.findall(r"N", smiles))
        amide_n = len(re.findall(r"NC\(=O\)|C\(=O\)N", smiles))
        basic_n = all_n - amide_n
        has_base = basic_n > 0

        if has_acid and has_base:
            return "zwitterion"
        if has_acid:
            return "acid"
        if has_base:
            return "base"
        return "neutral"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, smiles: str) -> TransporterProfile:
        """Predict transporter profile from SMILES string.

        Loads reference data and fits models on first call.

        Args:
            smiles: A valid SMILES string.

        Returns:
            TransporterProfile with per-transporter substrate/inhibitor
            probabilities, kinetic parameters, and DDI risk flags.
        """
        if self._ref_data is None:
            self._load_reference_data()
        mw = self._estimate_mw(smiles)
        logp = self._estimate_logp(smiles)
        charge = self._estimate_charge_class(smiles)
        return self._predict_from_features(mw, logp, charge, smiles)

    def predict_from_properties(
        self,
        mw: float,
        logp: float,
        charge_class: str = "neutral",
    ) -> TransporterProfile:
        """Predict transporter profile from known physicochemical properties.

        Args:
            mw: Molecular weight (g/mol).
            logp: Octanol-water partition coefficient.
            charge_class: One of "neutral", "acid", "base", "zwitterion".

        Returns:
            TransporterProfile with per-transporter predictions.
        """
        if self._ref_data is None:
            self._load_reference_data()
        return self._predict_from_features(mw, logp, charge_class, smiles=None)
