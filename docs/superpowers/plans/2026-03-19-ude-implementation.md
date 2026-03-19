# UDE Implementation Plan: Multi-Task End-to-End PK Learner

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an MLP to predict PK macro-parameters (F, Vd, ka, ke) end-to-end against observed Cmax, bypassing the error-accumulating ADME→IVIVE→ODE chain. Multi-task with CLint/fup auxiliary losses for regularization. Beat holdout AAFE 3.520 baseline.

**Architecture:**
```
SMILES → Morgan FP (2048) + RDKit (9)
       → Fingerprint encoder: Linear(2048, 64) → ReLU → Dropout(0.4)
       → Descriptor encoder:  Linear(9, 16)    → ReLU
       → Concat(80) → Linear(80, 32) → ReLU → Dropout(0.3)
       → PK Head:    Linear(32, 4) → constrained activations → (F, Vd, ka, ke)
       → CLint Head: Linear(32, 1) → exp → CLint (auxiliary)
       → fup Head:   Linear(32, 1) → sigmoid → fup (auxiliary)

PK Model: 1-compartment analytical formula
  tmax = ln(ka/ke) / (ka - ke)
  Cmax = F × dose × ka / (Vd × (ka-ke)) × [exp(-ke·tmax) - exp(-ka·tmax)]

Loss = w_quality × MSE_log(Cmax_pred, Cmax_obs)     [MMPK 1,098 drugs]
     + λ_clint × MSE_log(CLint_pred, CLint_obs)      [TDC 1,213 drugs]
     + λ_fup × MSE_log(fup_pred, fup_obs)             [TDC 1,614 drugs]
```

**Tech Stack:** PyTorch, RDKit (Morgan FP), scikit-learn (scaffold split), numpy

**Prerequisites:** UDE prerequisite gates ALL PASS (commit 4e07725)
- Holdout: `data/clinical/holdout_split.json` (71 drugs)
- Training: `data/ml/clinical/mmpk_quality_scored.csv` (1,098 clean drugs)
- Exclusions: `data/ml/clinical/mmpk_holdout_exclusions.json` (30 leaks)
- Baseline: AAFE 3.520 [2.57, 5.00] on holdout

**Why 1-compartment first (not full 35-state ODE):**
The pipeline's 30+ intermediate ADME parameters all reduce to 4 macro PK parameters
(F, Vd, ka, ke) that determine Cmax. Predicting these directly bypasses the entire
error-accumulating IVIVE chain. If this proof-of-concept fails, the full ODE will also
fail (same data, same features). If it succeeds, Phase 2 upgrades to the full ODE.

**Self-feedback iterations:** 5 rounds of critique resolved all CRITICAL/HIGH issues:
- P1/P2 (ODE complexity/stiffness): deferred to Phase 2
- P3 (overfitting): 134K params, 4K samples, dropout + auxiliary + early stopping
- P8 (singularity): torch.where + clamp
- P13 (bias init): physical defaults (F=0.5, Vd=70L, ka=1.0/h, ke=0.1/h)

---

## Phase 1: Analytical 1-Compartment Multi-Task Learner

### Task 1: Data Preprocessing Pipeline

**Files:**
- Create: `src/omega_pbpk/ml/models/ude/data.py`
- Create: `src/omega_pbpk/ml/models/ude/__init__.py`
- Test: `tests/ml/test_ude_data.py`

**Why:** Prepare training data from three sources with proper holdout exclusion,
quality weighting, and scaffold-split validation.

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_ude_data.py
import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def test_ude_data_module_exists():
    from omega_pbpk.ml.models.ude.data import UDEDataset
    assert UDEDataset is not None


def test_mmpk_dataset_loads():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training
    data = load_mmpk_training()
    assert len(data) >= 800
    assert "features" in data[0]
    assert "cmax_obs" in data[0]
    assert "quality_weight" in data[0]


def test_holdout_excluded():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training
    import json
    with open(REPO / "data" / "ml" / "clinical" / "mmpk_holdout_exclusions.json") as f:
        exclusions = json.load(f)
    excluded_names = set(exclusions["holdout_leaks_in_mmpk"])
    data = load_mmpk_training()
    train_names = {d["name"] for d in data}
    assert len(train_names & excluded_names) == 0


def test_scaffold_split():
    from omega_pbpk.ml.models.ude.data import load_mmpk_training, scaffold_split
    data = load_mmpk_training()
    train, val = scaffold_split(data, val_frac=0.2, seed=42)
    assert len(train) + len(val) == len(data)
    assert len(val) >= len(data) * 0.15  # at least 15% in val
```

- [ ] **Step 2: Implement data module**

```python
# src/omega_pbpk/ml/models/ude/data.py
"""Data loading for UDE multi-task training.

Three data sources:
1. MMPK (1,098 drugs): SMILES → observed Cmax (primary task)
2. TDC CLint (1,213): SMILES → observed CLint (auxiliary)
3. TDC fup (1,614): SMILES → observed fup (auxiliary)
"""
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def _smiles_to_features(smiles: str) -> np.ndarray | None:
    """Morgan FP (2048) + RDKit descriptors (9) = 2057 features."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.array(fp, dtype=np.float32)

    descs = np.array([
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol) / 200.0,
        Descriptors.MolWt(mol) / 600.0,
        Descriptors.NumHAcceptors(mol) / 10.0,
        Descriptors.NumHDonors(mol) / 5.0,
        Descriptors.NumRotatableBonds(mol) / 15.0,
        Descriptors.RingCount(mol) / 5.0,
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol) / 150.0,
    ], dtype=np.float32)

    return np.concatenate([fp_arr, descs])


def load_mmpk_training() -> list[dict]:
    """Load MMPK drugs for Cmax training, excluding holdout leaks."""
    quality_path = REPO / "data" / "ml" / "clinical" / "mmpk_quality_scored.csv"
    exclusion_path = REPO / "data" / "ml" / "clinical" / "mmpk_holdout_exclusions.json"

    with open(exclusion_path) as f:
        excluded = set(json.load(f)["holdout_leaks_in_mmpk"])

    data = []
    with open(quality_path) as f:
        for row in csv.DictReader(f):
            if row["name"] in excluded:
                continue
            if row["include"] != "True":
                continue

            features = _smiles_to_features(row["smiles"])
            if features is None:
                continue

            data.append({
                "name": row["name"],
                "smiles": row["smiles"],
                "features": features,
                "dose_mg": float(row["dose_mg"]),
                "cmax_obs": float(row["cmax_mg_L"]),
                "quality_weight": float(row["quality_score"]),
            })

    return data


def load_tdc_clint() -> list[dict]:
    """Load TDC Clearance_Hepatocyte_AZ for CLint auxiliary task."""
    try:
        from tdc.single_pred import ADME
        dataset = ADME(name="Clearance_Hepatocyte_AZ")
        df = dataset.get_data()
    except (ImportError, Exception):
        return []

    data = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        clint = float(row["Y"])
        if clint <= 0:
            continue
        features = _smiles_to_features(smiles)
        if features is None:
            continue
        data.append({
            "smiles": smiles,
            "features": features,
            "clint": clint,
        })
    return data


def load_tdc_fup() -> list[dict]:
    """Load TDC PPBR_AZ for fup auxiliary task."""
    try:
        from tdc.single_pred import ADME
        dataset = ADME(name="PPBR_AZ")
        df = dataset.get_data()
    except (ImportError, Exception):
        return []

    data = []
    for _, row in df.iterrows():
        smiles = str(row["Drug"])
        ppbr = float(row["Y"])
        fup = 1.0 - ppbr / 100.0
        if fup <= 0 or fup > 1:
            continue
        features = _smiles_to_features(smiles)
        if features is None:
            continue
        data.append({
            "smiles": smiles,
            "features": features,
            "fup": fup,
        })
    return data


def scaffold_split(data: list[dict], val_frac: float = 0.2, seed: int = 42):
    """Scaffold-based train/val split."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds = defaultdict(list)
    for i, d in enumerate(data):
        mol = Chem.MolFromSmiles(d["smiles"])
        if mol:
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                scaf = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(core))
            except Exception:
                scaf = d["smiles"]
        else:
            scaf = d["smiles"]
        scaffolds[scaf].append(i)

    rng = np.random.default_rng(seed)
    scaffold_list = list(scaffolds.values())
    rng.shuffle(scaffold_list)

    target_val = int(len(data) * val_frac)
    val_idx, train_idx = [], []
    for indices in scaffold_list:
        if len(val_idx) < target_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return [data[i] for i in train_idx], [data[i] for i in val_idx]
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ml/test_ude_data.py -v`
Expected: All PASS (TDC tests may skip if TDC not installed)

- [ ] **Step 4: Commit**

---

### Task 2: 1-Compartment Multi-Task Model

**Files:**
- Create: `src/omega_pbpk/ml/models/ude/model.py`
- Test: `tests/ml/test_ude_model.py`

**Why:** The core model — MLP encoder + PK/ADME heads + differentiable 1-cpt formula.

- [ ] **Step 1: Write failing tests**

```python
# tests/ml/test_ude_model.py
import torch
import pytest


def test_model_forward():
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel
    model = MultiTaskPKModel()
    x = torch.randn(4, 2057)  # batch of 4
    dose = torch.tensor([100.0, 200.0, 50.0, 400.0])
    cmax, pk_params = model.predict_cmax(x, dose)
    assert cmax.shape == (4,)
    assert all(c > 0 for c in cmax)


def test_pk_params_physical():
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel
    model = MultiTaskPKModel()
    x = torch.zeros(1, 2057)  # zero input → default bias values
    dose = torch.tensor([100.0])
    cmax, params = model.predict_cmax(x, dose)
    F, Vd, ka, ke = params["F"], params["Vd"], params["ka"], params["ke"]
    assert 0 < F.item() < 1, f"F={F.item()} out of range"
    assert 0.5 < Vd.item() < 5000, f"Vd={Vd.item()} out of range"
    assert 0.01 < ka.item() < 20, f"ka={ka.item()} out of range"
    assert 0.001 < ke.item() < 10, f"ke={ke.item()} out of range"


def test_cmax_formula_known():
    """Test 1-cpt formula against known analytical result."""
    from omega_pbpk.ml.models.ude.model import safe_cmax_1cpt
    # F=1, dose=100mg, Vd=100L, ka=2/h, ke=0.1/h
    # tmax = ln(2/0.1) / (2-0.1) = ln(20)/1.9 = 1.578h
    # Cmax = 100*2 / (100*1.9) * (exp(-0.1*1.578) - exp(-2*1.578))
    #      = 200/190 * (0.854 - 0.043) = 1.053 * 0.811 = 0.854 mg/L
    F = torch.tensor([1.0])
    dose = torch.tensor([100.0])
    Vd = torch.tensor([100.0])
    ka = torch.tensor([2.0])
    ke = torch.tensor([0.1])
    cmax = safe_cmax_1cpt(F, dose, Vd, ka, ke)
    assert abs(cmax.item() - 0.854) < 0.05, f"Cmax={cmax.item()}, expected ~0.854"


def test_gradient_flows():
    """Verify gradients flow through the 1-cpt formula."""
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel
    model = MultiTaskPKModel()
    x = torch.randn(2, 2057, requires_grad=False)
    dose = torch.tensor([100.0, 200.0])
    cmax, _ = model.predict_cmax(x, dose)
    loss = cmax.sum()
    loss.backward()
    # Check that encoder weights have gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum() > 0, f"No gradient for {name}"


def test_auxiliary_heads():
    from omega_pbpk.ml.models.ude.model import MultiTaskPKModel
    model = MultiTaskPKModel()
    x = torch.randn(3, 2057)
    clint = model.predict_clint(x)
    fup = model.predict_fup(x)
    assert clint.shape == (3,)
    assert all(c > 0 for c in clint)
    assert fup.shape == (3,)
    assert all(0 < f <= 1 for f in fup)
```

- [ ] **Step 2: Implement model**

```python
# src/omega_pbpk/ml/models/ude/model.py
"""Multi-task PK model: MLP encoder + 1-compartment analytical formula.

Architecture:
  Morgan FP (2048) → Linear(64) → ReLU → Dropout(0.4)
  RDKit (9) → Linear(16) → ReLU
  Concat(80) → Linear(32) → ReLU → Dropout(0.3)
  → PK Head (4): F, Vd, ka, ke via constrained activations
  → CLint Head (1): exp activation
  → fup Head (1): sigmoid activation
"""
import math

import torch
import torch.nn as nn


def safe_cmax_1cpt(
    F: torch.Tensor,
    dose: torch.Tensor,
    Vd: torch.Tensor,
    ka: torch.Tensor,
    ke: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable 1-compartment oral Cmax.

    Handles ka ≈ ke singularity via smooth switching.
    All inputs are (B,) tensors. Returns (B,) Cmax in mg/L.
    """
    Vd = Vd.clamp(min=0.5, max=5000.0)
    ka = ka.clamp(min=0.01, max=20.0)
    ke = ke.clamp(min=0.001, max=10.0)

    diff = ka - ke
    near_equal = diff.abs() < 0.01

    safe_diff = torch.where(near_equal, torch.full_like(diff, 0.01), diff)
    tmax = torch.log(ka / ke) / safe_diff
    tmax = tmax.clamp(min=0.01, max=72.0)

    cmax_std = (
        F * dose * ka / (Vd * safe_diff)
        * (torch.exp(-ke * tmax) - torch.exp(-ka * tmax))
    )
    cmax_deg = F * dose / (Vd * math.e)
    cmax = torch.where(near_equal, cmax_deg, cmax_std)
    return cmax.clamp(min=1e-10)


class MultiTaskPKModel(nn.Module):
    """Multi-task PK prediction model.

    Shared encoder with three output heads:
    - PK head: F, Vd, ka, ke → 1-cpt Cmax formula (primary)
    - CLint head: intrinsic clearance (auxiliary)
    - fup head: fraction unbound (auxiliary)
    """

    def __init__(self, fp_dim: int = 2048, desc_dim: int = 9):
        super().__init__()
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, 16),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(80, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # PK head: 4 outputs (F_logit, log_Vd, log_ka, log_ke)
        self.pk_head = nn.Linear(32, 4)

        # Auxiliary heads
        self.clint_head = nn.Linear(32, 1)  # log CLint
        self.fup_head = nn.Linear(32, 1)    # logit fup

        self._init_biases()

    def _init_biases(self):
        """Initialize PK head biases to physical defaults."""
        with torch.no_grad():
            # F: sigmoid(0) = 0.5
            self.pk_head.bias[0] = 0.0
            # Vd: exp(4.25) ≈ 70 L
            self.pk_head.bias[1] = math.log(70.0)
            # ka: exp(0) = 1.0 h⁻¹
            self.pk_head.bias[2] = 0.0
            # ke: exp(-2.3) ≈ 0.1 h⁻¹ (t½ ≈ 7h)
            self.pk_head.bias[3] = math.log(0.1)
            # CLint: exp(2) ≈ 7.4 µL/min/10^6 (moderate)
            self.clint_head.bias[0] = 2.0
            # fup: sigmoid(0) = 0.5
            self.fup_head.bias[0] = 0.0

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder: features → 32-dim latent."""
        fp = x[:, :2048]
        desc = x[:, 2048:]
        h_fp = self.fp_encoder(fp)
        h_desc = self.desc_encoder(desc)
        h = torch.cat([h_fp, h_desc], dim=1)
        return self.shared(h)

    def predict_cmax(
        self, x: torch.Tensor, dose: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Predict Cmax via 1-compartment model.

        Args:
            x: (B, 2057) molecular features
            dose: (B,) dose in mg

        Returns:
            cmax: (B,) predicted Cmax in mg/L
            params: dict with F, Vd, ka, ke tensors
        """
        h = self._encode(x)
        raw = self.pk_head(h)  # (B, 4)

        F = torch.sigmoid(raw[:, 0])          # (0, 1)
        Vd = torch.exp(raw[:, 1])             # (0, ∞) in L
        ka = torch.exp(raw[:, 2])             # (0, ∞) in h⁻¹
        ke = torch.exp(raw[:, 3])             # (0, ∞) in h⁻¹

        cmax = safe_cmax_1cpt(F, dose, Vd, ka, ke)

        return cmax, {"F": F, "Vd": Vd, "ka": ka, "ke": ke}

    def predict_clint(self, x: torch.Tensor) -> torch.Tensor:
        """Predict CLint (µL/min/10^6 cells)."""
        h = self._encode(x)
        return torch.exp(self.clint_head(h).squeeze(-1))

    def predict_fup(self, x: torch.Tensor) -> torch.Tensor:
        """Predict fraction unbound in plasma."""
        h = self._encode(x)
        return torch.sigmoid(self.fup_head(h).squeeze(-1))
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ml/test_ude_model.py -v`
Expected: All 5 tests PASS

- [ ] **Step 4: Commit**

---

### Task 3: Training Loop

**Files:**
- Create: `scripts/train_ude.py`
- Read: all data files from Task 1

**Why:** Multi-task training with quality-weighted Cmax loss, CLint/fup auxiliary losses,
scaffold-split validation, and early stopping.

- [ ] **Step 1: Implement training script**

```python
#!/usr/bin/env python3
"""Train UDE multi-task PK model.

Phase 1: 1-compartment analytical formula.
Multi-task: Cmax (MMPK 1,098) + CLint (TDC 1,213) + fup (TDC 1,614).

Usage:
    python scripts/train_ude.py
    python scripts/train_ude.py --epochs 200 --lr 5e-4
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.ml.models.ude.data import (  # noqa: E402
    load_mmpk_training,
    load_tdc_clint,
    load_tdc_fup,
    scaffold_split,
)
from omega_pbpk.ml.models.ude.model import MultiTaskPKModel  # noqa: E402


def compute_aafe(log_errors: torch.Tensor) -> float:
    """AAFE from log10 fold errors."""
    return float(10 ** log_errors.abs().mean())


def train(
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_clint: float = 0.1,
    lambda_fup: float = 0.1,
    batch_size: int = 64,
    patience: int = 15,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    mmpk_data = load_mmpk_training()
    print(f"  MMPK: {len(mmpk_data)} drugs")

    clint_data = load_tdc_clint()
    fup_data = load_tdc_fup()
    print(f"  TDC CLint: {len(clint_data)} drugs")
    print(f"  TDC fup: {len(fup_data)} drugs")

    # Scaffold split MMPK
    train_mmpk, val_mmpk = scaffold_split(mmpk_data, val_frac=0.2, seed=seed)
    print(f"  MMPK train: {len(train_mmpk)}, val: {len(val_mmpk)}")

    # Prepare tensors
    def to_tensors(data_list, keys):
        result = {}
        for key in keys:
            if key == "features":
                result[key] = torch.tensor(
                    np.stack([d[key] for d in data_list]), dtype=torch.float32
                ).to(device)
            else:
                result[key] = torch.tensor(
                    [d[key] for d in data_list], dtype=torch.float32
                ).to(device)
        return result

    train_t = to_tensors(train_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"])
    val_t = to_tensors(val_mmpk, ["features", "dose_mg", "cmax_obs", "quality_weight"])

    clint_t = to_tensors(clint_data, ["features", "clint"]) if clint_data else None
    fup_t = to_tensors(fup_data, ["features", "fup"]) if fup_data else None

    # Model
    model = MultiTaskPKModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = []

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        n_train = len(train_mmpk)

        # Shuffle
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            idx = perm[start:end]

            x = train_t["features"][idx]
            dose = train_t["dose_mg"][idx]
            cmax_obs = train_t["cmax_obs"][idx]
            weights = train_t["quality_weight"][idx]

            # Primary: Cmax loss (MSE in log10 space)
            cmax_pred, _ = model.predict_cmax(x, dose)
            log_err = torch.log10(cmax_pred.clamp(min=1e-10)) - torch.log10(cmax_obs.clamp(min=1e-10))
            loss_cmax = (weights * log_err ** 2).mean()

            loss = loss_cmax

            # Auxiliary: CLint
            if clint_t is not None and lambda_clint > 0:
                aux_idx = torch.randint(0, len(clint_data), (min(batch_size, len(clint_data)),), device=device)
                clint_pred = model.predict_clint(clint_t["features"][aux_idx])
                clint_obs = clint_t["clint"][aux_idx]
                loss_clint = ((torch.log10(clint_pred.clamp(min=1e-6)) - torch.log10(clint_obs.clamp(min=1e-6))) ** 2).mean()
                loss = loss + lambda_clint * loss_clint

            # Auxiliary: fup
            if fup_t is not None and lambda_fup > 0:
                aux_idx = torch.randint(0, len(fup_data), (min(batch_size, len(fup_data)),), device=device)
                fup_pred = model.predict_fup(fup_t["features"][aux_idx])
                fup_obs = fup_t["fup"][aux_idx]
                loss_fup = ((torch.log10(fup_pred.clamp(min=1e-6)) - torch.log10(fup_obs.clamp(min=1e-6))) ** 2).mean()
                loss = loss + lambda_fup * loss_fup

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_cmax.item() * (end - start)

        epoch_loss /= n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_cmax_pred, _ = model.predict_cmax(val_t["features"], val_t["dose_mg"])
            val_log_err = torch.log10(val_cmax_pred.clamp(min=1e-10)) - torch.log10(val_t["cmax_obs"].clamp(min=1e-10))
            val_loss = (val_t["quality_weight"] * val_log_err ** 2).mean().item()
            val_aafe = compute_aafe(val_log_err)

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_AAFE={val_aafe:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_aafe": round(val_aafe, 4),
        })

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Restore best model
    model.load_state_dict(best_state)

    # Save model
    model_dir = REPO / "models" / "ude"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_dir / "multitask_pk_phase1.pt")

    meta = {
        "phase": 1,
        "model": "1-compartment analytical",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_val_aafe": round(history[best_epoch - 1]["val_aafe"], 4),
        "n_train_mmpk": len(train_mmpk),
        "n_val_mmpk": len(val_mmpk),
        "n_tdc_clint": len(clint_data),
        "n_tdc_fup": len(fup_data),
        "n_params": n_params,
        "hyperparams": {
            "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
            "lambda_clint": lambda_clint, "lambda_fup": lambda_fup,
            "batch_size": batch_size, "patience": patience,
        },
        "history": history,
    }
    with open(model_dir / "meta_phase1.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nModel saved to {model_dir}")
    print(f"Best epoch: {best_epoch}, val AAFE: {meta['best_val_aafe']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-clint", type=float, default=0.1)
    parser.add_argument("--lambda-fup", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    train(
        epochs=args.epochs, lr=args.lr,
        lambda_clint=args.lambda_clint, lambda_fup=args.lambda_fup,
        batch_size=args.batch_size, patience=args.patience,
    )
```

- [ ] **Step 2: Run training**

Run: `python scripts/train_ude.py --epochs 150 --lr 1e-3`
Expected: val AAFE decreasing, best around epoch 50-80, val AAFE ~2.0-3.0

- [ ] **Step 3: Commit**

---

### Task 4: Holdout Evaluation

**Files:**
- Create: `scripts/evaluate_ude.py`
- Create: `outputs/ude_phase1_holdout.json`

**Why:** THE test. Compare UDE Phase 1 holdout AAFE to pipeline baseline 3.520.

- [ ] **Step 1: Implement evaluation**

```python
#!/usr/bin/env python3
"""Evaluate UDE model on permanent holdout set.

Compares to pipeline baseline AAFE 3.520 [2.57, 5.00].

Usage:
    python scripts/evaluate_ude.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from omega_pbpk.ml.models.ude.data import _smiles_to_features  # noqa: E402
from omega_pbpk.ml.models.ude.model import MultiTaskPKModel  # noqa: E402


def bootstrap_aafe_ci(fold_errors, n_boot=10000, seed=42):
    log_fe = np.log10(np.array(fold_errors))
    rng = np.random.default_rng(seed)
    n = len(log_fe)
    boots = [float(10 ** np.mean(np.abs(log_fe[rng.integers(0, n, n)]))) for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    # Load model
    model_path = REPO / "models" / "ude" / "multitask_pk_phase1.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model = MultiTaskPKModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load holdout
    with open(REPO / "data" / "clinical" / "holdout_split.json") as f:
        split = json.load(f)
    with open(REPO / "data" / "clinical" / "platinum_reference.json") as f:
        plat = json.load(f)

    holdout_drugs = split["holdout"]
    results = []
    fold_errors = []

    print(f"Evaluating UDE Phase 1 on {len(holdout_drugs)} holdout drugs")
    print("=" * 70)

    for drug_name in sorted(holdout_drugs):
        entry = plat["drugs"].get(drug_name)
        if entry is None:
            continue

        features = _smiles_to_features(entry["smiles"])
        if features is None:
            print(f"  SKIP {drug_name}: features failed")
            continue

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        dose = torch.tensor([entry["dose_mg"]], dtype=torch.float32)

        with torch.no_grad():
            cmax_pred, params = model.predict_cmax(x, dose)

        pred = cmax_pred.item()
        obs = entry["cmax_mg_L"]
        fe = max(pred / obs, obs / pred) if pred > 0 and obs > 0 else float("nan")

        if not np.isnan(fe):
            fold_errors.append(fe)

        results.append({
            "drug": drug_name,
            "pred_cmax": round(pred, 6),
            "obs_cmax": round(obs, 6),
            "fold_error": round(fe, 4) if not np.isnan(fe) else None,
            "F": round(params["F"].item(), 4),
            "Vd": round(params["Vd"].item(), 2),
            "ka": round(params["ka"].item(), 4),
            "ke": round(params["ke"].item(), 4),
        })

        symbol = "ok" if fe <= 2.0 else ("~" if fe <= 3.0 else "X")
        print(f"  {symbol} {drug_name:25s} FE={fe:6.2f}x  pred={pred:.4f}  obs={obs:.4f}  "
              f"F={params['F'].item():.2f} Vd={params['Vd'].item():.0f}")

    # Metrics
    valid_fe = [fe for fe in fold_errors if not np.isnan(fe)]
    log_fe = np.log10(valid_fe)
    aafe = float(10 ** np.mean(np.abs(log_fe)))
    pct_2fold = sum(1 for fe in valid_fe if fe <= 2.0) / len(valid_fe) * 100
    ci_lo, ci_hi = bootstrap_aafe_ci(valid_fe)

    # Compare to baseline
    baseline_aafe = 3.520
    improvement = (baseline_aafe - aafe) / baseline_aafe * 100

    output = {
        "model": "UDE Phase 1 (1-compartment)",
        "n_holdout": len(holdout_drugs),
        "n_evaluated": len(valid_fe),
        "aafe": round(aafe, 4),
        "ci95_lo": round(ci_lo, 4),
        "ci95_hi": round(ci_hi, 4),
        "pct_2fold": round(pct_2fold, 1),
        "baseline_aafe": baseline_aafe,
        "improvement_pct": round(improvement, 1),
        "per_drug": results,
    }

    print(f"\n{'=' * 70}")
    print(f"UDE Phase 1:  AAFE = {aafe:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]  %2-fold = {pct_2fold:.1f}%")
    print(f"Baseline:     AAFE = {baseline_aafe:.3f} [2.567, 4.997]")
    print(f"Improvement:  {improvement:+.1f}%")
    if aafe < baseline_aafe:
        print(">>> UDE BEATS BASELINE")
    else:
        print(">>> UDE DOES NOT BEAT BASELINE")

    out_path = REPO / "outputs" / "ude_phase1_holdout.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation**

Run: `python scripts/evaluate_ude.py`
Expected: AAFE < 3.5 (must beat or match baseline 3.520)

- [ ] **Step 3: Decision gate**

| Holdout AAFE | Action |
|-------------|--------|
| < 2.5 | Phase 1 이미 strong → Phase 2 optional |
| 2.5 - 3.0 | Phase 2 진행 (35-state ODE upgrade) |
| 3.0 - 3.5 | Hyperparameter tuning / architecture search |
| > 3.5 | Phase 1 failed → feature engineering 또는 더 큰 dataset 필요 |

- [ ] **Step 4: Commit**

---

## Phase 2: Full 35-State Differentiable ODE (Conditional)

**Proceed only if Phase 1 holdout AAFE < 3.0.**

### Task 5: Differentiable ODE Wrapper

**Files:**
- Create: `src/omega_pbpk/ml/models/ude/differentiable_pbpk.py`
- Test: `tests/ml/test_ude_ode.py`

**Why:** Replace 1-cpt analytical formula with full 35-state PBPK ODE via torchdiffeq.
The encoder predicts ADME micro-parameters → ODE computes Cmax mechanistically.

**Key design decisions:**
- Port `body.py:_rhs()` to PyTorch — vectorize by pre-computing organ indices
- Remove if/else branching → hard-code liver (idx 7), kidney (idx 5), gut_wall (idx 8) by index
- Use `torchdiffeq.odeint(method='dopri5', rtol=1e-5, atol=1e-7)`
- Batch dimension: solve multiple drugs in parallel
- ADME head: 6 outputs (δ_CLint_hep, δ_CLint_gut, δ_fup, δ_logP_eff, δ_pKa_eff, δ_ka)
- Kp[14 tissues] via Berezhkovskiy formula (differentiable, using logP_eff + pKa_eff)

**Estimated LOC:** ~300 lines (ODE) + 100 lines (ADME head) + 100 lines (tests)

This task's full implementation depends on Phase 1 results and is intentionally
left as a high-level spec rather than detailed code.

- [ ] **Step 1: Verify torchdiffeq is installed**

Run: `pip install torchdiffeq`

- [ ] **Step 2: Port _rhs() to PyTorch**

Key transformation:
```python
# body.py (numpy, branching):
if name == "liver":
    clh = q * fup * clint / (q + fup * clint)

# differentiable_pbpk.py (PyTorch, vectorized):
# Pre-compute: liver_idx = 7, liver_flow = Q_liver
clh = liver_flow * fup * clint / (liver_flow + fup * clint)
dydt[:, LIVER_IDX] = liver_flow * c_in - liver_flow * c_out - clh * c_out
```

- [ ] **Step 3: Validate forward pass matches scipy**

Run both solvers on 10 test drugs. Assert max relative error < 5%.

- [ ] **Step 4: Replace 1-cpt with ODE in training loop**

- [ ] **Step 5: Retrain + evaluate on holdout**

---

### Task 6: Pipeline Integration

**Files:**
- Modify: `src/omega_pbpk/pipeline/__init__.py`
- Create: `src/omega_pbpk/ml/models/ude/predictor.py`

**Why:** Make UDE available as an alternative Cmax predictor alongside MetaLearner.

- [ ] **Step 1: Create UDE predictor wrapper**

```python
# src/omega_pbpk/ml/models/ude/predictor.py
class UDEPredictor:
    """UDE-based Cmax predictor for pipeline integration."""

    def __init__(self, model_path=None):
        self.model = MultiTaskPKModel()
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def predict(self, smiles: str, dose_mg: float) -> float:
        features = _smiles_to_features(smiles)
        if features is None:
            return None
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        dose = torch.tensor([dose_mg], dtype=torch.float32)
        with torch.no_grad():
            cmax, _ = self.model.predict_cmax(x, dose)
        return float(cmax.item())
```

- [ ] **Step 2: Add feature flag to pipeline**

Add `_USE_UDE = False` flag. When True, use UDE prediction as additional blend input
to the MetaLearner (or replace it).

- [ ] **Step 3: Run accuracy regression tests**

Run: `pytest tests/ml/test_accuracy_regression.py -v`
Expected: PASS (UDE is opt-in, default pipeline unchanged)

- [ ] **Step 4: Commit**

---

## Success Criteria

| Metric | Baseline | Phase 1 Target | Phase 2 Target |
|--------|----------|----------------|----------------|
| Holdout AAFE | 3.520 | < 3.0 | < 2.5 |
| Holdout %2-fold | 52.1% | > 55% | > 65% |
| Val AAFE (MMPK) | — | < 2.5 | < 2.0 |
| Training time | — | < 30 min | < 2 hours |
| Regression tests | PASS | PASS | PASS |

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Overfitting (134K params, 1,098 drugs) | Auxiliary tasks, dropout(0.3-0.4), weight decay, early stopping |
| 1-cpt too simplistic | Phase 2 ODE upgrade; 1-cpt is proof of concept only |
| Gradient vanishing in ODE (Phase 2) | gradient clipping, dopri5→implicit_adams fallback |
| TDC data not available | Graceful fallback (auxiliary loss disabled) |
| Holdout has prodrugs (4 drugs) | Flag as OOD; report AAFE with/without prodrugs |
