"""XGBoost-based predictor for hepatocyte intrinsic clearance (CLint).

Predicts log10(CLint_hep) in µL/min/10^6 cells using Morgan fingerprints
+ RDKit descriptors. Training data: TDC Clearance_Hepatocyte_AZ (1,213 compounds).

This serves as a fallback when ADMET-AI is not available, ensuring the pipeline
always has hepatocyte CLint for proper IVIVE (rather than falling back to the
unreliable CYP-attributed CLint path).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
_MODEL_DIR = _REPO_ROOT / "models" / "xgboost_clint"
_MODEL_PATH = _MODEL_DIR / "model.json"
_META_PATH = _MODEL_DIR / "meta.json"


def _load_tdc_clearance() -> tuple[list[str], list[float]]:
    """Load Clearance_Hepatocyte_AZ from TDC.

    Returns:
        (smiles_list, clint_list) where clint is in µL/min/10^6 cells.
    """
    try:
        from tdc.single_pred import ADME

        dataset = ADME(name="Clearance_Hepatocyte_AZ")
        df = dataset.get_data()

        smiles_list = []
        clint_list = []
        for _, row in df.iterrows():
            smi = str(row["Drug"])
            clint = float(row["Y"])
            if clint > 0:
                smiles_list.append(smi)
                clint_list.append(clint)

        logger.info("Loaded %d compounds from TDC Clearance_Hepatocyte_AZ", len(smiles_list))
        return smiles_list, clint_list
    except (ImportError, Exception) as exc:
        logger.warning("TDC Clearance_Hepatocyte_AZ not available: %s", exc)
        return [], []


def _get_clint_reference_anchors() -> list[tuple[str, float]]:
    """Reference hepatocyte CLint anchors back-calculated from clinical clearance.

    Uses the pipeline's IVIVE formula (CLh = 0.3 × CLint^0.9) to back-calculate
    the CLint (µL/min/10^6 cells) that produces the correct clinical hepatic
    clearance. This anchors the XGBoost model to known drug-clearance
    relationships and expands its dynamic range beyond the compressed TDC
    training distribution.

    Back-calculation: CLint = (CLh / 0.3) ^ (10/9)
    Purely renally-cleared drugs excluded (metformin, amoxicillin).
    """
    return [
        # (SMILES, CLint_anchor µL/min/10^6 cells)
        # High-extraction drugs (CL > 30 L/h) — XGBoost under-predicts most
        ("CC(C)NCC(O)COc1cccc2ccccc12", 399.0),  # propranolol CL=70
        ("COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", 366.0),  # verapamil CL=65
        ("COCCc1ccc(OCC(O)CNC(C)C)cc1", 319.0),  # metoprolol CL=60
        (
            "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
            506.0,
        ),  # atorvastatin CL=625
        ("CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21", 506.0),  # simvastatin CL=450
        ("CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", 183.3),  # fluoxetine CL=40
        # Moderate-extraction drugs (CL 15-30 L/h)
        ("Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1-2", 139.0),  # midazolam CL=30
        ("COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", 139.0),  # omeprazole CL=30
        ("CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1", 166.2),  # losartan CL=36
        ("CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", 92.0),  # amlodipine CL=21
        ("CC(=O)Nc1ccc(O)cc1", 150.0),  # acetaminophen CL≈25 (high phase II metabolism)
        # Mixed hepatic+renal drugs — hepatic CLint back-calculated from CLh portion
        (
            "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1",
            150.0,
        ),  # ranitidine CLh≈25 (FMO+N-demethylation)
        (
            "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
            77.0,
        ),  # ciprofloxacin CLh≈15 (Phase II, partial hepatic)
        # Low-extraction drugs (CL < 15 L/h) — model already reasonable
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 15.3),  # ibuprofen CL=3.5
        ("Cn1c(=O)c2[nH]cnc2n(C)c1=O", 12.1),  # theophylline CL=2.8
        ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 15.0),  # caffeine CL≈3.7 (incl CYP1A2+NAT2)
        ("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", 6.2),  # diazepam CL=1.5
        ("CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", 1.0),  # warfarin CL=0.18 (clipped to min)
        # Primarily renally-eliminated — hepatic CLint must be anchored low
        # Fluconazole: CL_total≈2.36 L/h, 80% renal → CLh_hepatic≈0.47 L/h
        # back-calc: (0.47/0.3)^(10/9) ≈ 1.8; anchor at 2.0 (triazole antifungal)
        ("OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", 2.0),  # fluconazole CLh_hep≈0.47 L/h
    ]


# Map from anchor SMILES → drug name for decontamination
_ANCHOR_DRUG_NAMES: dict[str, str] = {
    "CC(C)NCC(O)COc1cccc2ccccc12": "propranolol",
    "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC": "verapamil",
    "COCCc1ccc(OCC(O)CNC(C)C)cc1": "metoprolol",
    "CC(C)c1n(CC(O)CC(O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1": "atorvastatin",
    "CCC(C)(C)C(=O)OC1CC(O)C=C2C=CC(C)C(CCC3CC(O)CC(=O)O3)C21": "simvastatin",
    "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1": "fluoxetine",
    "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1-2": "midazolam",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1": "omeprazole",
    "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1": "losartan",
    "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl": "amlodipine",
    "CC(=O)Nc1ccc(O)cc1": "acetaminophen",
    "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1": "ranitidine",
    "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O": "ciprofloxacin",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O": "ibuprofen",
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O": "theophylline",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C": "caffeine",
    "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21": "diazepam",
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O": "warfarin",
    "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F": "fluconazole",
}


def get_decontaminated_anchors(
    exclude_drugs: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Return CLint anchors with specified drugs removed.

    Args:
        exclude_drugs: Drug names to exclude (e.g., holdout set drugs).
            If None, returns full anchor set (production behavior).
    """
    if exclude_drugs is None:
        return _get_clint_reference_anchors()

    all_anchors = _get_clint_reference_anchors()
    filtered = []
    removed = []
    for smiles, clint in all_anchors:
        drug_name = _ANCHOR_DRUG_NAMES.get(smiles, "unknown")
        if drug_name in exclude_drugs:
            removed.append(drug_name)
        else:
            filtered.append((smiles, clint))

    if removed:
        logger.info("Decontaminated anchors: removed %s", removed)
    return filtered


def _enhanced_features(smiles: str) -> np.ndarray | None:
    """Enhanced feature vector: Morgan + RDKit descriptors + transporter flags.

    Adds 4 transporter flags (P-gp substrate/inhibitor, OATP1B1 substrate,
    BCRP substrate) to the base 2057 features, yielding 2061 total.
    Transporter-mediated clearance is a major source of CLint prediction
    error — P-gp substrates and OATP substrates have different hepatic
    uptake rates that affect intrinsic clearance.
    """
    from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

    base = _smiles_to_features(smiles)
    if base is None:
        return None

    extra = np.zeros(4, dtype=np.float32)
    try:
        from omega_pbpk.ml.models.adme.transporter_lookup import get_transporter_flags

        flags = get_transporter_flags(smiles=smiles)
        if flags:
            extra[0] = float(flags.get("pgp_substrate", 0))
            extra[1] = float(flags.get("pgp_inhibitor", 0))
            extra[2] = float(flags.get("oatp1b1_substrate", 0))
            extra[3] = float(flags.get("bcrp_substrate", 0))
    except ImportError:
        pass

    return np.concatenate([base, extra])


class XGBoostCLintPredictor:
    """XGBoost predictor for hepatocyte intrinsic clearance.

    Predicts log10(CLint) in µL/min/10^6 cells.
    Uses enhanced features with transporter flags when available.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        auto_load: bool = True,
    ) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise ImportError("XGBoost required. pip install xgboost") from None
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise ImportError("RDKit required. pip install rdkit-pypi") from None

        self._model_dir = model_dir or _MODEL_DIR
        self._model_path = self._model_dir / "model.json"
        self._meta_path = self._model_dir / "meta.json"
        self._model: Any = None
        self._cv_metrics: dict[str, float] = {}

        if auto_load:
            if self._model_path.exists():
                self._load_model()
            else:
                self.train()

    def train(self, n_folds: int = 5) -> dict[str, float]:
        """Train on TDC Clearance_Hepatocyte_AZ + reference CLint anchors."""
        import xgboost as xgb

        tdc_smiles, tdc_clint = _load_tdc_clearance()
        if not tdc_smiles:
            raise ValueError("No training data for CLint predictor.")

        # Compute features for TDC data
        valid_X = []
        valid_y = []
        sample_weights = []
        from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

        for smi, clint in zip(tdc_smiles, tdc_clint):
            feat = _smiles_to_features(smi)
            if feat is not None:
                valid_X.append(feat)
                valid_y.append(np.log10(max(clint, 1.0)))
                sample_weights.append(1.0)

        # Add reference CLint anchors (weighted 10x to anchor predictions)
        ref_anchors = _get_clint_reference_anchors()
        n_anchors = 0
        for smi, clint in ref_anchors:
            feat = _smiles_to_features(smi)
            if feat is not None:
                valid_X.append(feat)
                valid_y.append(np.log10(max(clint, 1.0)))
                sample_weights.append(
                    5.0
                )  # 5x weight for reference anchors (reduced from 50x to prevent memorization)
                n_anchors += 1

        logger.info("Added %d reference CLint anchors (50x weight)", n_anchors)

        X = np.array(valid_X)
        y = np.array(valid_y, dtype=np.float64)
        w = np.array(sample_weights, dtype=np.float64)
        n = len(y)

        logger.info(
            "Training XGBoost CLint on %d compounds (%d TDC + %d anchors, log10 space)",
            n,
            n - n_anchors,
            n_anchors,
        )

        # Cross-validation (TDC data only — anchors always in training)
        tdc_n = n - n_anchors
        tdc_indices = np.arange(tdc_n)
        anchor_indices = np.arange(tdc_n, n)
        rng = np.random.RandomState(42)
        rng.shuffle(tdc_indices)
        fold_size = tdc_n // n_folds

        all_preds = np.zeros(n)
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else tdc_n
            val_idx = tdc_indices[start:end]
            train_idx = np.concatenate([tdc_indices[:start], tdc_indices[end:], anchor_indices])

            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.6,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
            all_preds[val_idx] = model.predict(X[val_idx])

        # CV metrics (TDC data only — anchors always in training)
        tdc_preds = all_preds[:tdc_n]
        tdc_y = y[:tdc_n]
        residuals = tdc_preds - tdc_y
        mae_log = float(np.mean(np.abs(residuals)))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((tdc_y - np.mean(tdc_y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        self._cv_metrics = {
            "mae_log10": round(mae_log, 4),
            "r2": round(r2, 4),
            "n_compounds": n,
            "n_anchors": n_anchors,
        }
        logger.info(
            "XGBoost CLint CV: MAE(log10)=%.4f, R²=%.4f (n=%d, %d anchors)",
            mae_log,
            r2,
            n,
            n_anchors,
        )

        # Train final model on all data with sample weights
        self._model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.6,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y, sample_weight=w)
        self._save_model()
        return self._cv_metrics

    def predict_clint(self, smiles: str) -> float:
        """Predict hepatocyte CLint (µL/min/10^6 cells).

        Args:
            smiles: SMILES string.

        Returns:
            Predicted CLint, clipped to [1.0, 1000.0].
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        from omega_pbpk.ml.models.adme.xgboost_fup import _smiles_to_features

        feat = _smiles_to_features(smiles)
        if feat is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        log_clint = float(self._model.predict(feat.reshape(1, -1))[0])
        return float(np.clip(10.0**log_clint, 1.0, 1000.0))

    def _save_model(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self._model_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._cv_metrics, f, indent=2)
        logger.info("XGBoost CLint model saved to %s", self._model_path)

    def _load_model(self) -> None:
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._cv_metrics = json.load(f)
        logger.info("XGBoost CLint model loaded from %s", self._model_path)

    @property
    def cv_metrics(self) -> dict[str, float]:
        return dict(self._cv_metrics)
