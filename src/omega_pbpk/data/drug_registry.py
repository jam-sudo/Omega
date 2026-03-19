"""Canonical drug registry for benchmarking.

Single source of truth for benchmark drug SMILES and doses.
Imported by benchmark scripts and regression tests.
"""

from __future__ import annotations

# All benchmark drugs: name -> {smiles, dose_mg, set?}
# First 20 = dev set, last 5 = validation set
BENCHMARK_DRUGS: dict[str, dict] = {
    "caffeine": {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "dose_mg": 100},
    "metoprolol": {"smiles": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "dose_mg": 100},
    "midazolam": {"smiles": "Clc1ccc2c(c1)C(=NCc1nccn1C)c1ccccc1N2", "dose_mg": 2},
    "propranolol": {"smiles": "CC(C)NCC(O)COc1cccc2ccccc12", "dose_mg": 80},
    "warfarin": {"smiles": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "dose_mg": 10},
    "d_amphetamine": {"smiles": "C[C@@H](N)Cc1ccccc1", "dose_mg": 20},
    "ibuprofen": {"smiles": "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "dose_mg": 400},
    "acetaminophen": {"smiles": "CC(=O)Nc1ccc(O)cc1", "dose_mg": 1000},
    "amoxicillin": {"smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "dose_mg": 500},
    "atorvastatin": {
        "smiles": "CC(C)c1n(CC[C@@H](O)C[C@@H](O)CC(=O)O)c(-c2ccccc2)c(-c2ccc(F)cc2)c1C(=O)Nc1ccccc1",
        "dose_mg": 40,
    },
    "carbamazepine": {"smiles": "NC(=O)N1c2ccccc2C=Cc2ccccc21", "dose_mg": 200},
    "diazepam": {"smiles": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "dose_mg": 10},
    "digoxin": {
        "smiles": "C[C@@H]1O[C@@H](O[C@@H]2C[C@H](O)[C@@H](O[C@@H]3C[C@H](O)[C@@H](O[C@@H]4C[C@H](O)[C@@H](OC5CC(CO)=CC(=O)O5)C(C)O4)C(C)O3)C(C)O2)C[C@H](O)[C@H]1O",
        "dose_mg": 0.5,
    },
    "fluoxetine": {"smiles": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1", "dose_mg": 20},
    "nifedipine": {"smiles": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c1ccccc1[N+](=O)[O-]", "dose_mg": 10},
    "omeprazole": {"smiles": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "dose_mg": 20},
    "phenytoin": {"smiles": "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1", "dose_mg": 300},
    "theophylline": {"smiles": "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "dose_mg": 300},
    "verapamil": {"smiles": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", "dose_mg": 80},
    "tramadol": {"smiles": "COc1cccc(C2(O)CCCCC2CN(C)C)c1", "dose_mg": 100},
    # --- Validation set ---
    "atenolol": {"smiles": "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1", "dose_mg": 50, "set": "validation"},
    "fluconazole": {
        "smiles": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
        "dose_mg": 200,
        "set": "validation",
    },
    "furosemide": {
        "smiles": "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl",
        "dose_mg": 40,
        "set": "validation",
    },
    "gabapentin": {"smiles": "OC(=O)CC1(CN)CCCCC1", "dose_mg": 300, "set": "validation"},
    "metformin": {"smiles": "CN(C)C(=N)NC(=N)N", "dose_mg": 500, "set": "validation"},
}

# Core-24 drugs used in gold24_reference_cmax.json
# (all BENCHMARK_DRUGS except tramadol, which was added later)
CORE24_NAMES: frozenset[str] = frozenset(BENCHMARK_DRUGS.keys()) - {"tramadol"}


def get_core24() -> dict[str, dict]:
    """Return only the core-24 benchmark drugs."""
    return {k: v for k, v in BENCHMARK_DRUGS.items() if k in CORE24_NAMES}
