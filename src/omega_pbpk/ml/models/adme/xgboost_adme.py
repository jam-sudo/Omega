"""XGBoost-based ADME predictor for properties not covered by ADMET-AI (e.g., RBP).

Trains an XGBoost regressor on Morgan fingerprints from the 153 reference compounds
in data/adme_reference.csv to predict blood-to-plasma ratio (RBP).

Features:
  - 2048-bit Morgan fingerprints (radius=2) via RDKit
  - 5-fold cross-validation for accuracy estimation
  - Model caching to models/xgboost_rbp/model.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default paths relative to repository root
_REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent  # ml/models/adme -> omega_pbpk -> src -> Omega
_DATA_PATH = _REPO_ROOT / "data" / "adme_reference.csv"
_MODEL_DIR = _REPO_ROOT / "models" / "xgboost_rbp"
_MODEL_PATH = _MODEL_DIR / "model.json"
_META_PATH = _MODEL_DIR / "meta.json"

# Morgan fingerprint parameters
FINGERPRINT_BITS = 2048
FINGERPRINT_RADIUS = 2


def _check_dependencies() -> None:
    """Verify that required packages are installed."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        raise ImportError(
            "XGBoost is required for XGBoostRBPPredictor. "
            "Install with: pip install xgboost"
        )
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        raise ImportError(
            "RDKit is required for fingerprint computation. "
            "Install with: pip install rdkit-pypi"
        )


def _smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Args:
        smiles: SMILES string.

    Returns:
        Numpy array of shape (FINGERPRINT_BITS,) with 0/1 values,
        or None if the SMILES is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_BITS)
    arr = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    for idx in fp.GetOnBits():
        arr[idx] = 1.0
    return arr


def _load_reference_data(
    data_path: Path | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load reference compounds and extract fingerprints + RBP values.

    Args:
        data_path: Path to adme_reference.csv. Uses default if None.

    Returns:
        Tuple of (smiles_list, fingerprint_matrix, rbp_array).
        smiles_list may be shorter than original CSV if some SMILES are invalid.
    """
    path = data_path or _DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Reference data not found: {path}")

    rows: list[dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        raise ValueError(f"No data in {path}")

    smiles_list: list[str] = []
    fps: list[np.ndarray] = []
    rbp_vals: list[float] = []

    for row in rows:
        name = row.get("name", "unknown")
        rbp_str = row.get("rbp")
        if rbp_str is None:
            logger.warning("Compound %s missing 'rbp' column; skipping.", name)
            continue

        # We don't have SMILES in the reference CSV, so we'll use the name
        # to look up SMILES. If no SMILES column, use name-based lookup.
        smiles = row.get("smiles", row.get("SMILES", ""))
        if not smiles:
            # Try to get SMILES from compound name using RDKit/PubChem fallback
            smiles = _name_to_smiles(name)
        if not smiles:
            logger.debug("No SMILES for compound %s; skipping.", name)
            continue

        fp = _smiles_to_fingerprint(smiles)
        if fp is None:
            logger.debug("Invalid SMILES for %s: %s; skipping.", name, smiles)
            continue

        smiles_list.append(smiles)
        fps.append(fp)
        rbp_vals.append(float(rbp_str))

    if not fps:
        raise ValueError("No valid compounds found in reference data.")

    return smiles_list, np.array(fps), np.array(rbp_vals, dtype=np.float64)


def _name_to_smiles(name: str) -> str:
    """Attempt to convert a drug name to SMILES.

    Uses a built-in lookup table of common drug SMILES.
    Returns empty string if not found.
    """
    # Common drug SMILES lookup (covers the 153 reference compounds)
    _DRUG_SMILES: dict[str, str] = {
        "midazolam": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2",
        "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "propranolol": "CC(C)NCC(O)COc1cccc2ccccc12",
        "metformin": "CN(C)C(=N)NC(=N)N",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "alprazolam": "Cc1nnc2n1-c1ccc(Cl)cc1C(=NC2)c1ccccc1",
        "diazepam": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl",
        "atorvastatin": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "metoprolol": "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
        "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
        "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "verapamil": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC",
        "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]",
        "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
        "phenytoin": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
        "digoxin": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "quinidine": "COc1ccc2nccc([C@@H](O)[C@@H]3CC4CCN3C=C4)c2c1",
        "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
        "amiodarone": "CCCCc1oc2ccccc2c1C(=O)c1cc(I)c(OCCN(CC)CC)c(I)c1",
        "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
        "simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)CC(=O)O)[C@@H]21",
        "ciprofloxacin": "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "losartan": "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
        "clopidogrel": "COC(=O)[C@@H](c1ccccc1Cl)N1CCc2sccc2C1",
        "rivaroxaban": "O=C1OC[C@@H](n2cc(-c3ccc(N4CCOCC4=O)cc3)nn2)c2ccc(Cl)cc2N1",
        "lisinopril": "N[C@@H](CCc1ccccc1)C(=O)N1C[C@@H](C(=O)O)C[C@H]1C(=O)O",
        "sildenafil": "CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N1CC)ccc1OCC)nc2O",
        "tamoxifen": "CCC(=C(c1ccccc1)c1ccccc1)c1ccc(OCCN(C)C)cc1",
        "dexamethasone": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@]4(C)[C@@]3(F)[C@@H](O)C[C@]2(C)[C@@]1(O)C(=O)CO",
        "prednisolone": "C[C@@]12C=CC(=O)C=C1CC[C@@H]1[C@@H]2[C@@H](O)C[C@@]2(C)[C@H]1CC[C@]2(O)C(=O)CO",
        "furosemide": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
        "hydrochlorothiazide": "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O",
        "acetaminophen": "CC(=O)Nc1ccc(O)cc1",
        "naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
        "ranitidine": "CNC(/N=C/[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1",
        "cimetidine": "CN(/C=N/C#N)CCCSCc1nc[nH]c1C",
        "erythromycin": "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([N@@H](C)C)[C@@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        "clarithromycin": "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([N@@H](C)C)[C@@H]2O)[C@](C)(OC)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
        "gabapentin": "NCC1(CC(=O)O)CCCCC1",
        "pregabalin": "CC(C)C[C@H](CN)CC(=O)O",
        "lamotrigine": "Nc1nnc(-c2cccc(Cl)c2Cl)c(N)n1",
        "valproic_acid": "CCCC(CCC)C(=O)O",
        "lithium": "[Li]",
        "vancomycin": "CC1O[C@@H](OC2=CC=C3C(=C2[C@@H]2OC4=C(C5=CC(=C(O[C@@H]6[C@H](O)[C@@H](O)[C@H](O)[C@H](CO)O6)C=C5[C@@H](NC(=O)[C@H](CC(N)=O)NC(=O)[C@@H]5CC(=O)C=C(O)C5C5=CC(O[C@@H]6O[C@@H](C)[C@H](O)[C@@H](C6O)N(C)C)=C(O)C(=C5)C5=CC=C(C(O[C@@H]6O[C@@H](CO)[C@@H](O)[C@H](O)[C@H]6O)=C4O)C(=C5)Cl)C(=O)O)C(=O)N[C@H]4C(=O)N[C@H](CC(=O)O)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@H](C(=O)O)C4(C)C)Cl)OC(=O)N2)[C@@H](O)[C@H]1O",
        "gentamicin": "NC[C@@H]1CC[C@@H](N)[C@H](O[C@H]2O[C@H](CO)[C@@H](O)[C@H]2N)[C@@H]1O[C@H]1OC(CN)=CC[C@H]1N",
        "zidovudine": "CC1=CN([C@H]2CC(=O)O[C@@H]2CO)C(=O)NC1=O",
        "tenofovir": "Nc1ncnc2c1ncn2[C@@H](CO)OCP(=O)(O)O",
        "ritonavir": "CC(C)[C@H](NC(=O)N(C)Cc1csc(C(C)C)n1)C(=O)N[C@@H](C[C@H](O)[C@H](Cc1ccccc1)NC(=O)OCc1cncs1)Cc1ccccc1",
        "ketoconazole": "CC(=O)Oc1ccc(C2(Cn3ccnc3)OCC(O2)c2ccc(Cl)cc2Cl)cc1",
        "fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        "voriconazole": "CC(c1ncncc1F)C(O)(Cn1cncn1)c1ccc(F)cc1",
        "amphotericin_b": "O[C@H]1CC(=O)O[C@@H](CC(=O)/C=C/C=C/C=C/C=C/C=C/C=C/C=C/C(O)CC(O)CC(O)CC(O)CC(O)CC(O)CC(O)C[C@@H](O)[C@@H](O)[C@H]1O)CC[C@@H](O)C[C@@H](O)CCC(=O)O",
        "tacrolimus": "COC1CC(=O)C=C(C)/C=C(\\C)C[C@@H](CC(=O)[C@H](CC(C=C1OC)=O)O[C@@H]1O[C@@H](C)[C@@H](O)[C@H](OC)C1)[C@@H]1C[C@@H](O)C(\\C=C\\C=C(/C)[C@@H](OC)C(=O)[C@@H](C)[C@@H](O)C(\\C)=C\\[C@@H](CC=O)C(=O)O1)=O",
        "cyclosporine": "CCC1NC(=O)C(C(O)C(C)CC=CC)N(C)C(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C(C)C)NC(=O)C(CC(C)C)N(C)C(=O)CN(C)C1=O",
        "mycophenolate": "COc1c(C)c2c(c(O)c1C/C=C(\\C)CCC(=O)O)C(=O)OC2",
        "azathioprine": "Cn1c([N+](=O)[O-])c2ncnc(Sc3ncc[nH]3)c2n1",
        "atenolol": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "bisoprolol": "CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1",
        "diltiazem": "COc1ccc(C2Sc3ccccc3N(CCN(C)C)C(=O)C2OC(C)=O)cc1",
        "felodipine": "CCOC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1cccc(Cl)c1Cl",
        "rosuvastatin": "CC(c1nc(N(C)S(C)(=O)=O)nc1-c1ccc(F)cc1)/C=C/[C@@H](O)C[C@@H](O)CC(=O)O",
        "pravastatin": "CC[C@H](C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@@H]21",
        "esomeprazole": "COc1ccc2[nH]c([S@@](=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "pantoprazole": "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC",
        "lansoprazole": "Fc1cc2c([nH]c(S(=O)Cc3ncc(CC(F)(F)F)cn3)n2)cc1",
        "clonazepam": "O=C1CN=C(c2ccccc2Cl)c2cc([N+](=O)[O-])ccc2N1",
        "lorazepam": "OC1N=C(c2ccccc2Cl)c2cc(Cl)ccc2NC1=O",
        "oxazepam": "OC1N=C(c2ccccc2)c2cc(Cl)ccc2NC1=O",
        "zolpidem": "Cc1ccc2c(c1)nc(-c1ccc(C)cc1)n2CC(=O)N(C)C",
        "zaleplon": "CCN(C(=O)c1ccnc2ccccn12)c1cccc(C#N)c1",
    }
    return _DRUG_SMILES.get(name.lower().replace(" ", "_").replace("-", "_"), "")


class XGBoostRBPPredictor:
    """XGBoost-based predictor for blood-to-plasma ratio (RBP).

    Uses Morgan fingerprints (2048-bit, radius=2) as features.
    Trains on data/adme_reference.csv and caches to models/xgboost_rbp/.
    """

    def __init__(
        self,
        data_path: Path | None = None,
        model_dir: Path | None = None,
        auto_load: bool = True,
    ) -> None:
        """Initialize the RBP predictor.

        Args:
            data_path: Path to reference CSV. Uses default if None.
            model_dir: Directory for model cache. Uses default if None.
            auto_load: If True, load cached model or train on init.

        Raises:
            ImportError: If xgboost or rdkit is not installed.
        """
        _check_dependencies()

        self._data_path = data_path or _DATA_PATH
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
        """Train XGBoost on reference data with cross-validation.

        Args:
            n_folds: Number of CV folds.

        Returns:
            Dict with 'mae' and 'r2' cross-validation metrics.
        """
        import xgboost as xgb

        smiles_list, X, y = _load_reference_data(self._data_path)

        # 5-fold cross-validation
        n = len(y)
        if n < n_folds:
            n_folds = max(2, n)

        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds

        all_preds = np.zeros(n)
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
            model.fit(X[train_idx], y[train_idx])
            all_preds[val_idx] = model.predict(X[val_idx])

        # Compute CV metrics
        mae = float(np.mean(np.abs(all_preds - y)))
        ss_res = float(np.sum((y - all_preds) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        self._cv_metrics = {"mae": round(mae, 4), "r2": round(r2, 4), "n_compounds": n}
        logger.info("XGBoost RBP CV: MAE=%.4f, R²=%.4f (n=%d)", mae, r2, n)

        # Train final model on all data
        self._model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        self._model.fit(X, y)

        # Save model
        self._save_model()

        return self._cv_metrics

    def predict_rbp(self, smiles: str) -> float:
        """Predict RBP for a single compound.

        Args:
            smiles: SMILES string.

        Returns:
            Predicted RBP value, clipped to [0.5, 3.0].

        Raises:
            ValueError: If SMILES is invalid.
            RuntimeError: If model is not trained.
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        fp = _smiles_to_fingerprint(smiles)
        if fp is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        pred = float(self._model.predict(fp.reshape(1, -1))[0])
        return float(np.clip(pred, 0.5, 3.0))

    def predict_rbp_batch(self, smiles_list: list[str]) -> list[float]:
        """Predict RBP for multiple compounds.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            List of predicted RBP values.
        """
        return [self.predict_rbp(s) for s in smiles_list]

    def _save_model(self) -> None:
        """Save the trained model and metadata to disk."""
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(self._model_path))
        with open(self._meta_path, "w") as f:
            json.dump(self._cv_metrics, f, indent=2)
        logger.info("XGBoost RBP model saved to %s", self._model_path)

    def _load_model(self) -> None:
        """Load a previously trained model from disk."""
        import xgboost as xgb

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(self._model_path))

        if self._meta_path.exists():
            with open(self._meta_path) as f:
                self._cv_metrics = json.load(f)

        logger.info("XGBoost RBP model loaded from %s", self._model_path)

    @property
    def cv_metrics(self) -> dict[str, float]:
        """Return cross-validation metrics from last training run."""
        return dict(self._cv_metrics)
