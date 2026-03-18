#!/usr/bin/env python3
"""Train 6 binary XGBoost classifiers for transporter substrate prediction.

Strategy:
  1. P-gp: Use TDC Pgp_Broccatelli dataset (1218 entries) if available,
     otherwise fall back to reference CSV
  2. OATP1B1, BCRP, OCT2, OAT, PepT1: Use data/transporter_reference.csv
     (95 entries) with SMILES resolved from reference_database.json or
     hardcoded fallback

Each classifier: SMILES -> 2057 features (Morgan FP 2048 + 9 descriptors)
                         -> XGBoost binary classifier
                         -> P(substrate)

Models saved to: models/corrections/transporters/<name>_classifier.json
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MODEL_DIR = REPO_ROOT / "models" / "corrections" / "transporters"
TRANSPORTER_REF_CSV = REPO_ROOT / "data" / "transporter_reference.csv"
REF_DB_PATH = REPO_ROOT / "data" / "clinical" / "reference_database.json"

# Mapping from transporter column name in CSV to our short names
TRANSPORTER_COLUMNS = {
    "pgp": "pgp_substrate",
    "oatp1b1": "oatp1b1_substrate",
    "bcrp": "bcrp_substrate",
    "oct2": "oct2_substrate",
    "oat": "mrp2_substrate",  # OAT1/3 proxied by MRP2 (renal drug transporter marker)
    "pept1": "mate1_substrate",  # PepT1 proxied by MATE1 (renal peptide/drug transporter)
}

# Hardcoded SMILES for common drugs not in the reference database.
# These are canonical SMILES from PubChem.
_EXTRA_SMILES = {
    "ritonavir": "CC(C)[C@@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)C(=O)N[C@H](C[C@H](O)[C@@H](CC1=CC=CC=C1)NC(=O)OCC1=CC=CC=C1)CC1=CC=CC=C1",
    "ketoconazole": "O=C1N(C2CCC(OC3=CC=C(OC[C@@H]4OC(CN5C=CN=C5)(C5=CC=C(Cl)CC5)O4)C=C3)CC2)C(=O)CS1",
    "quinidine": "C=C[C@H]1CN2CC[C@@H]1C[C@@H]2[C@H](O)C1=CC=NC2=CC=C(OC)C=C12",
    "digoxin": "O[C@@H]1[C@@H](O)[C@H](OC2CC[C@@]3(C)[C@H](CC[C@@H]4[C@@H]3C[C@@H](O)[C@]3(C)C(=CC4)C4=CC(=O)OC4)[C@H]2O)O[C@@H]1C.O[C@@H]1[C@@H](O)[C@H](O)O[C@@H]1C",
    "fexofenadine": "CC(C)(C1=CC=C(C=C1)C(CCCN2CCC(CC2)C(C3=CC=CC=C3)(C4=CC=CC=C4)O)O)C(=O)O",
    "dabigatran": "CC1=CC(=C(N1CC2=CC=C(C=C2)C(=N)N)C)/N=N/C3=CC4=CC=CC=C4C(=C3)C(=O)N(CC)CC",
    "simvastatin": "CC[C@H](C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@@H](C)[C@H](CC[C@@H](O)CC(=O)O)[C@@H]21",
    "lovastatin": "CC[C@H](C)C(=O)O[C@H]1C[C@@H](O)C=C2C=C[C@@H](C)[C@H](CC[C@@H](O)CC(=O)O)[C@@H]21",
    "cimetidine": "CC1=C(N=CN1)CSCCNC(=NC)NC#N",
    "trimethoprim": "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N",
    "dolutegravir": "C[C@H]1CCO[C@@H]2N1C(=O)C3=C(C(=O)C(=CN3C2)C(=O)NCC4=CC(=CC=C4)F)O",
    "prazosin": "COC1=CC2=C(C=C1OC)C(=NC(=N2)N3CCN(CC3)C(=O)C4=CC=CO4)N",
    "nitrofurantoin": "O=C1CN(/N=C/C2=CC=C([N+]([O-])=O)O2)C(=O)N1",
    "lopinavir": "CC1=CC=CC=C1C[C@@H](C(=O)N[C@H](CC2=CC=CC=C2)[C@@H](CC3=CC=CC=C3)NC(=O)[C@@H](C(C)C)N4CCCNC4=O)O",
    "gemfibrozil": "CC1=CC(=C(C=C1)OCCCC(C)(C)C(=O)O)C",
    "warfarin": "CC(=O)C[C@@H](C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "bosentan": "COCCOC1=CC2=C(C=C1)N=C(N=C2NS(=O)(=O)C1=CC=C(C=C1)C(C)(C)C)C1=CC=CC=C1OC",
    "olmesartan": "CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NN=N[NH]4)C(=O)O)C(C)(C)O",
    "candesartan": "CCOC(=O)OC1=NC2=CC=CC=C2N1CC3=CC=C(C=C3)C4=CC=CC=C4C5=NN=N[NH]5",
    "repaglinide": "CCOC(=O)C1=CC=CC=C1NC(=O)CC(CC2CCCCC2)C3=CC=C(C=C3)CC(C)C",
    "erythromycin": "CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O",
    "sirolimus": "CO[C@@H]1C[C@@H](CC(=O)[C@H](/C=C(/[C@@H]([C@H](/C=C/C=C/C(=O)[C@H](CC(/C=C/[C@@H](C(=O)[C@@H](OC(=O)[C@H]2CCN3CCCCC23)[C@H](C)C[C@@H]1OC)C)O)OC)O)C)OC)C)C",
    "imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
    "dexamethasone": "C[C@@H]1C[C@H]2[C@@H]3CCC4=CC(=O)C=C[C@@]4([C@]3([C@H](C[C@@]2([C@]1(C(=O)CO)O)C)O)F)C",
    "amitriptyline": "CN(C)CCC=C1C2=CC=CC=C2CCC3=CC=CC=C31",
    "nortriptyline": "CNCC/C=C\\1/C2=CC=CC=C2CCC3=CC=CC=C31",
    "imipramine": "CN(C)CCCN1C2=CC=CC=C2CCC3=CC=CC=C31",
    "haloperidol": "OC1(CCN(CCCC(=O)C2=CC=C(F)C=C2)CC1)C1=CC=C(Cl)C=C1",
    "atenolol": "CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1",
    "bisoprolol": "CC(C)NCC(O)COC1=CC=C(COCCOC(C)C)C=C1",
    "methadone": "CCC(=O)C(CC1=CC=CC=C1)(CC1=CC=CC=C1)C(C)N(C)C",
    "fentanyl": "CCC(=O)N(C1CCN(CCC2=CC=CC=C2)CC1)C1=CC=CC=C1",
    "oxycodone": "CN1CC[C@]23C4=C(C=C(C=C4)O)O[C@@H]2[C@@H](CC1[C@@H]3O)OC(=O)C",
    "furosemide": "OC(=O)C1=CC(=CC(NCC2=CC=CO2)=C1Cl)S(N)(=O)=O",
    "hydrochlorothiazide": "NS(=O)(=O)C1=CC2=C(NCNS2(=O)=O)C=C1Cl",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C1=CC=CC=C1Cl)C(=O)OC",
    "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C1C1=CC=CC=C1[N+]([O-])=O)C(=O)OC",
    "lansoprazole": "CC1=C(C=CN=C1CS(=O)C2=NC3=CC=CC=C3N2)OCC(F)(F)F",
    "phenytoin": "O=C1NC(=O)C(C2=CC=CC=C2)(C2=CC=CC=C2)N1",
    "edoxaban": "CN1CC(CC1C(=O)NC1CCC(CC1)N1C(=O)C2=CC(Cl)=CC=C2NC1=O)NC(=O)C(NC(=O)C1=NC=C(C(=O)O)S1)C1CCCC1",
    "tenofovir": "NC1=C2N=CN([C@@H]3CO[C@@H](CS3)OCP(=O)(O)O)C2=NC=N1",
    "naloxone": "O=C(C=C1C2)C3=C1O[C@@H]4[C@@]5(O)[C@H]2[C@@H](O)C[C@]3([C@H]4N(CC=C)C5)O",
    "buprenorphine": "CO[C@]12CC[C@]3(O)[C@H]4CC5=CC=C(O)C6=C5[C@@]3(CCN4CC4CC4)[C@H]1C6=C(C(C)(C)C)C2=O",
    "glipizide": "CC1=CN=C(C=C1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
    "glyburide": "COC1=CC=C(C=C1)Cl.CCC1=CC=C(C=C1)C(=O)NCCC2=CC=C(C=C2)S(=O)(=O)NC(=O)NC3CCCCC3",
    "cetirizine": "OC(=O)COCCN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
    "loratadine": "CCOC(=O)N1CCC(=C2C3=CC=C(Cl)C=C3CCC4=CC=CC=N42)CC1",
}


def _load_ref_db_smiles() -> dict[str, str]:
    """Load drug name -> SMILES mapping from the reference database."""
    mapping: dict[str, str] = {}
    if not REF_DB_PATH.exists():
        return mapping
    with open(REF_DB_PATH) as fh:
        db = json.load(fh)
    for name, info in db.get("drugs", {}).items():
        smiles = info.get("smiles")
        if smiles:
            mapping[name.lower()] = smiles
    return mapping


def _load_transporter_reference() -> list[dict]:
    """Load transporter reference CSV rows."""
    rows = []
    if not TRANSPORTER_REF_CSV.exists():
        logger.warning("Transporter reference CSV not found: %s", TRANSPORTER_REF_CSV)
        return rows
    with open(TRANSPORTER_REF_CSV, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name", "").strip()
            if name:
                rows.append(row)
    return rows


def _extract_features(smiles: str) -> np.ndarray | None:
    """Extract 2057-dim feature vector from SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint (2048 bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.zeros(2048, dtype=np.float32)
    for idx in fp.GetOnBits():
        fp_arr[idx] = 1.0

    # 9 physicochemical descriptors (normalized)
    descs = np.array(
        [
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol) / 200.0,
            Descriptors.MolWt(mol) / 600.0,
            Descriptors.NumHAcceptors(mol) / 10.0,
            Descriptors.NumHDonors(mol) / 5.0,
            Descriptors.NumRotatableBonds(mol) / 15.0,
            Descriptors.RingCount(mol) / 5.0,
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol) / 150.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([fp_arr, descs])


def _load_tdc_pgp() -> tuple[np.ndarray, np.ndarray] | None:
    """Try loading TDC Pgp_Broccatelli dataset. Returns (X, y) or None."""
    try:
        from tdc.single_pred import ADME

        data = ADME(name="Pgp_Broccatelli")
        df = data.get_data()
        logger.info("TDC Pgp_Broccatelli: %d entries loaded", len(df))
    except Exception as exc:
        logger.warning("TDC not available: %s", exc)
        return None

    X_list = []
    y_list = []
    skipped = 0
    for _, row in df.iterrows():
        features = _extract_features(row["Drug"])
        if features is not None:
            X_list.append(features)
            y_list.append(int(row["Y"]))
        else:
            skipped += 1

    if skipped > 0:
        logger.info("TDC P-gp: skipped %d entries (invalid SMILES)", skipped)

    if len(X_list) < 10:
        logger.warning("TDC P-gp: too few valid entries (%d)", len(X_list))
        return None

    return np.array(X_list), np.array(y_list)


def _load_reference_data(
    transporter_col: str, name_to_smiles: dict[str, str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load training data for a transporter from reference CSV.

    Returns (X, y) or None if insufficient data.
    """
    rows = _load_transporter_reference()
    if not rows:
        return None

    X_list = []
    y_list = []
    skipped = 0

    for row in rows:
        name = row["name"].strip().lower()
        smiles = name_to_smiles.get(name) or _EXTRA_SMILES.get(name)
        if smiles is None:
            skipped += 1
            continue

        features = _extract_features(smiles)
        if features is None:
            skipped += 1
            continue

        label = int(row.get(transporter_col, 0))
        X_list.append(features)
        y_list.append(label)

    if skipped > 0:
        logger.info(
            "Reference CSV (%s): skipped %d entries (no SMILES or invalid)",
            transporter_col,
            skipped,
        )

    if len(X_list) < 5:
        logger.warning("Reference CSV (%s): too few entries (%d)", transporter_col, len(X_list))
        return None

    return np.array(X_list), np.array(y_list)


def _train_classifier(X: np.ndarray, y: np.ndarray, name: str, n_pos: int, n_neg: int):
    """Train a single XGBoost binary classifier with class balancing."""
    import xgboost as xgb

    # Handle class imbalance with scale_pos_weight
    scale_pos_weight = max(n_neg / max(n_pos, 1), 1.0)

    model = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=50,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)

    # Quick training stats
    preds = model.predict_proba(X)[:, 1]
    from sklearn.metrics import roc_auc_score

    try:
        auc = roc_auc_score(y, preds)
        logger.info(
            "  %s: N=%d (pos=%d, neg=%d), train AUC=%.3f",
            name,
            len(y),
            n_pos,
            n_neg,
            auc,
        )
    except ValueError:
        # Only one class present
        logger.info(
            "  %s: N=%d (pos=%d, neg=%d), train AUC=N/A (single class)",
            name,
            len(y),
            n_pos,
            n_neg,
        )

    return model


def main() -> None:
    """Train all 6 transporter classifiers and save to disk."""

    logger.info("=" * 60)
    logger.info("Training Transporter Substrate Classifiers")
    logger.info("=" * 60)

    # Create output directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load SMILES mappings
    name_to_smiles = _load_ref_db_smiles()
    # Add hardcoded fallback SMILES
    for name, smiles in _EXTRA_SMILES.items():
        if name not in name_to_smiles:
            name_to_smiles[name] = smiles

    logger.info("SMILES available for %d drugs", len(name_to_smiles))

    trained = 0

    for transporter_name, csv_col in TRANSPORTER_COLUMNS.items():
        logger.info("\n--- %s (column: %s) ---", transporter_name.upper(), csv_col)

        # Special handling for P-gp: use TDC if available
        if transporter_name == "pgp":
            tdc_data = _load_tdc_pgp()
            if tdc_data is not None:
                X, y = tdc_data
                n_pos = int(y.sum())
                n_neg = len(y) - n_pos
                logger.info("Using TDC data for P-gp (%d entries)", len(y))
            else:
                ref_data = _load_reference_data(csv_col, name_to_smiles)
                if ref_data is None:
                    logger.warning("No data available for %s — skipping", transporter_name)
                    continue
                X, y = ref_data
                n_pos = int(y.sum())
                n_neg = len(y) - n_pos
                logger.info("Using reference CSV for P-gp (%d entries)", len(y))
        else:
            ref_data = _load_reference_data(csv_col, name_to_smiles)
            if ref_data is None:
                logger.warning("No data available for %s — skipping", transporter_name)
                continue
            X, y = ref_data
            n_pos = int(y.sum())
            n_neg = len(y) - n_pos

        # Need at least 1 positive and 1 negative example
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Skipping %s: only one class present (pos=%d, neg=%d)",
                transporter_name,
                n_pos,
                n_neg,
            )
            continue

        model = _train_classifier(X, y, transporter_name, n_pos, n_neg)

        # Save model
        model_path = MODEL_DIR / f"{transporter_name}_classifier.json"
        model.save_model(str(model_path))
        logger.info("  Saved: %s", model_path)
        trained += 1

    logger.info("\n" + "=" * 60)
    logger.info("Done: %d/%d classifiers trained", trained, len(TRANSPORTER_COLUMNS))
    logger.info("Models saved to: %s", MODEL_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
