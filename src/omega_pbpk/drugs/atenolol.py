"""Atenolol — beta-blocker, renally cleared, hydrophilic base, low lipophilicity.

Literature references:
  - Kirch W et al., Clin Pharmacokinet, 1981 (PK parameters)
  - Wan SH et al., Clin Pharmacol Ther, 1979 (renal clearance)
  - Mason WD et al., Clin Pharmacol Ther, 1979 (oral bioavailability)

Key PK properties:
  - Low plasma protein binding: fup = 0.95 (only 3-5% bound)
  - Moderate Vss ~ 63-95 L (hydrophilic, distributes to total body water)
  - Primarily renally cleared (~90% excreted unchanged): CLr ~ 10 L/h
  - Minimal hepatic metabolism: negligible CYP contribution
  - Low oral bioavailability: F ~ 40-60% (incomplete absorption)
  - Low intestinal permeability (BCS Class III)
  - t1/2 ~ 6-7 h
"""

from omega_pbpk.drugs.drug import Drug

ATENOLOL = Drug(
    name="Atenolol",
    mw=266.34,
    logP=0.16,
    pka=[9.6],
    drug_type="monoprotic_base",
    compound_type="base",
    partition_method="heuristic",
    fup=0.95,
    rbp=1.0,
    smiles="CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
    # In vitro CLint — negligible (renally cleared drug)
    clint={
        "CYP3A4": 0.0,
        "CYP2D6": 0.0,
    },
    fm={
        "other": 1.0,
    },
    # Calibrated in vivo CLint (L/h) — minimal hepatic metabolism
    clint_hepatic_L_per_h=1.0,
    clint_gut_L_per_h=0.0,
    # Renal clearance (L/h) — dominant elimination route (~90%)
    clr_L_per_h=10.0,
    # Absorption: BCS Class III — low permeability, high solubility
    peff=0.4,
    solubility_mg_mL=26.5,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — near-unity for well-perfused, lower for lipophilic tissues
    kp={
        "lung": 0.8,
        "brain": 0.15,
        "heart": 0.9,
        "kidney": 1.2,
        "liver": 0.7,
        "spleen": 0.6,
        "gut_wall": 0.8,
        "pancreas": 0.5,
        "thymus": 0.4,
        "reproductive": 0.5,
        "rest": 0.5,
    },
    # Permeability-limited organs
    permeability_limited={
        "adipose": {"kp": 0.1, "ps": 5.0},
        "muscle": {"kp": 0.5, "ps": 15.0},
        "bone": {"kp": 0.3, "ps": 3.0},
        "skin": {"kp": 0.4, "ps": 5.0},
    },
    gut_clint_multiplier=0.0,
)
