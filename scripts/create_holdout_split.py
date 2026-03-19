#!/usr/bin/env python3
"""Create permanent scaffold-stratified holdout split from platinum 147.

Strategy:
1. Extract Murcko generic scaffolds for all 147 drugs
2. Cluster drugs by scaffold (same scaffold → same split)
3. Force all tuning_contaminated drugs into training set
4. Greedily assign remaining scaffold clusters to train/holdout
   targeting 55/45 split, balancing data_quality tiers
5. Save to data/clinical/holdout_split.json (PERMANENT — never regenerate)

Usage:
    python scripts/create_holdout_split.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PLATINUM_PATH = REPO / "data" / "clinical" / "platinum_reference.json"
OUTPUT_PATH = REPO / "data" / "clinical" / "holdout_split.json"
SEED = 42


def get_generic_scaffold(smiles: str) -> str:
    """Extract Murcko generic scaffold from SMILES."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"UNPARSEABLE_{smiles[:20]}"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic)
    except Exception:
        return f"SCAFFOLD_ERROR_{smiles[:20]}"


def main():
    if OUTPUT_PATH.exists():
        print(f"ERROR: {OUTPUT_PATH} already exists. This is a PERMANENT split.")
        print("Delete manually with --force if you truly need to regenerate.")
        if "--force" not in sys.argv:
            sys.exit(1)

    # Load platinum
    with open(PLATINUM_PATH) as f:
        plat = json.load(f)
    drugs = plat["drugs"]
    print(f"Loaded {len(drugs)} platinum drugs")

    # Extract scaffolds
    scaffold_map = {}  # drug_name -> scaffold_smiles
    scaffold_clusters = defaultdict(list)  # scaffold -> [drug_names]
    for name, entry in drugs.items():
        scaffold = get_generic_scaffold(entry["smiles"])
        scaffold_map[name] = scaffold
        scaffold_clusters[scaffold].append(name)

    n_scaffolds = len(scaffold_clusters)
    print(f"Found {n_scaffolds} unique Murcko scaffolds")

    # Identify contaminated drugs (must go to train)
    contaminated = {
        name for name, entry in drugs.items() if entry.get("tuning_contaminated", False)
    }
    print(f"Contaminated drugs (forced to train): {len(contaminated)}")

    # Identify scaffolds that MUST be in train (contain contaminated drugs)
    forced_train_scaffolds = set()
    for scaffold, members in scaffold_clusters.items():
        if any(m in contaminated for m in members):
            forced_train_scaffolds.add(scaffold)

    # Remaining scaffolds for splitting
    free_scaffolds = [
        (scaffold, members)
        for scaffold, members in scaffold_clusters.items()
        if scaffold not in forced_train_scaffolds
    ]

    # Shuffle free scaffolds deterministically
    random.seed(SEED)
    random.shuffle(free_scaffolds)

    # Greedy assignment targeting 45% holdout of total
    target_holdout = int(0.45 * len(drugs))
    forced_train_drugs = set()
    for scaffold in forced_train_scaffolds:
        forced_train_drugs.update(scaffold_clusters[scaffold])

    holdout = []
    train = list(forced_train_drugs)

    for _scaffold, members in free_scaffolds:
        if len(holdout) + len(members) <= target_holdout + 5:
            holdout.extend(members)
        else:
            train.extend(members)

    # Sort for reproducibility
    train.sort()
    holdout.sort()

    # Validate
    assert set(train) | set(holdout) == set(drugs.keys())
    assert set(train) & set(holdout) == set()
    assert contaminated.issubset(set(train))

    # Quality tier distribution
    for split_name, split_drugs in [("train", train), ("holdout", holdout)]:
        tiers = defaultdict(int)
        for d in split_drugs:
            tiers[drugs[d].get("data_quality", "unknown")] += 1
        print(f"\n{split_name} ({len(split_drugs)} drugs):")
        for tier, count in sorted(tiers.items()):
            print(f"  {tier}: {count}")

    # Save
    result = {
        "train": train,
        "holdout": holdout,
        "metadata": {
            "n_train": len(train),
            "n_holdout": len(holdout),
            "split_method": "murcko_generic_scaffold_stratified",
            "seed": SEED,
            "n_scaffolds": n_scaffolds,
            "forced_train_contaminated": len(contaminated),
            "scaffold_assignments": scaffold_map,
            "created": "2026-03-19",
            "WARNING": "PERMANENT SPLIT — do not regenerate. Hold-out drugs must never be used for training, tuning, or threshold selection.",
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Train: {len(train)}, Holdout: {len(holdout)}")


if __name__ == "__main__":
    main()
