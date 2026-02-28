"""Ibuprofen — NSAID, CYP2C9 substrate, high protein binding, acid.

Literature references:
  - Davies NM, Clin Pharmacokinet, 1998 (comprehensive PK review)
  - Mazaleuskaya LL et al., J Clin Pharmacol, 2015 (pharmacogenomics)
  - Rudy AC et al., J Pharmacol Exp Ther, 1991 (CYP2C9 metabolism)

Key PK properties:
  - High plasma protein binding: fup = 0.01 (99% bound to albumin)
  - Low Vss ~ 7-10 L (reflecting high protein binding)
  - Moderate hepatic clearance: CL ~ 3-4 L/h, t1/2 ~ 1.5-3 h
  - Primary metabolic route: CYP2C9 (60-80%), CYP2C8 (minor)
  - Near-complete oral bioavailability: F ~ 80-100%
  - Negligible renal clearance of unchanged drug
"""

from omega_pbpk.drugs.drug import Drug

IBUPROFEN = Drug(
    name="Ibuprofen",
    mw=206.28,
    logP=3.97,
    pka=[4.91],
    drug_type="monoprotic_acid",
    compound_type="acid",
    partition_method="heuristic",
    fup=0.01,
    rbp=0.55,
    smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    # In vitro CLint (reference values)
    clint={
        "CYP2C9": 6.0,
        "CYP2C8": 2.0,
    },
    fm={
        "CYP2C9": 0.75,
        "CYP2C8": 0.20,
        "other": 0.05,
    },
    # Calibrated in vivo CLint (L/h) — retrograde from clinical PK
    # CL_plasma ~ 3-4 L/h, t1/2 ~ 1.5-3 h (Davies 1998)
    clint_hepatic_L_per_h=350.0,
    clint_gut_L_per_h=0.0,
    clr_L_per_h=0.0,
    # Absorption: BCS Class II — high permeability, low solubility
    peff=3.0,
    solubility_mg_mL=0.021,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — low values reflecting very high protein binding (fup=0.01)
    kp={
        "lung": 0.05,
        "brain": 0.04,
        "heart": 0.08,
        "kidney": 0.12,
        "liver": 0.30,
        "spleen": 0.08,
        "gut_wall": 0.15,
        "pancreas": 0.07,
        "thymus": 0.05,
        "reproductive": 0.06,
        "rest": 0.07,
    },
    # Permeability-limited organs
    permeability_limited={
        "adipose": {"kp": 0.04, "ps": 3.0},
        "muscle": {"kp": 0.06, "ps": 8.0},
        "bone": {"kp": 0.04, "ps": 2.0},
        "skin": {"kp": 0.05, "ps": 4.0},
    },
    gut_clint_multiplier=0.1,
)
