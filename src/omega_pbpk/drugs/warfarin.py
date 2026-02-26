"""Warfarin — anticoagulant, CYP2C9 substrate, high protein binding (acid).

Literature references:
  - Breckenridge A et al., Clin Pharmacokinet, 1985
  - Holford NHG, Clin Pharmacokinet, 1986 (Vss ~ 9-10 L, t1/2 ~ 37 h)
  - Nishimura RA et al., J Am Coll Cardiol, 2014 (clinical PK)
  - Takahashi H & Echizen H, Clin Pharmacokinet, 2001 (CYP2C9 metabolism)

Key PK properties:
  - Very high plasma protein binding: fup = 0.005 (99.5% bound to albumin)
  - Low hepatic clearance: CL_plasma ~ 0.18 L/h, t1/2 ~ 37 h
  - Near-complete oral bioavailability: F ~ 99%
  - Narrow therapeutic window (INR 2-3)
  - Primary metabolic route: CYP2C9 (S-warfarin); CYP3A4 minor (R-warfarin)
  - No significant renal or gut-wall elimination
"""

from omega_pbpk.drugs.drug import Drug

WARFARIN = Drug(
    name="Warfarin",
    mw=308.33,
    logP=2.7,
    pka=[5.1],
    drug_type="monoprotic_acid",
    compound_type="acid",
    partition_method="heuristic",
    fup=0.005,
    rbp=0.58,
    smiles="OC(CC(=O)c1ccccc1)c1c(=O)oc2ccccc12",
    # In vitro CLint (reference values; calibrated in vivo CLint used below)
    clint={
        "CYP2C9": 0.18,
        "CYP3A4": 0.04,
    },
    fm={
        "CYP2C9": 0.80,
        "CYP3A4": 0.15,
        "other": 0.05,
    },
    # Calibrated in vivo CLint (L/h) — retrograde from clinical PK
    # CL_plasma ~ 0.18 L/h, t1/2 ~ 37 h (Holford 1986)
    clint_hepatic_L_per_h=15.0,
    clint_gut_L_per_h=0.0,
    clr_L_per_h=0.0,
    # Absorption: BCS Class I — high permeability, low solubility (at neutral pH)
    peff=2.5,
    solubility_mg_mL=0.054,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — perfusion-limited organs
    # Low Kp values reflect high protein binding (low fup = 0.005)
    kp={
        "lung": 0.2,
        "brain": 0.05,
        "heart": 0.15,
        "kidney": 0.25,
        "liver": 0.80,
        "spleen": 0.20,
        "gut_wall": 0.30,
        "pancreas": 0.18,
        "thymus": 0.12,
        "reproductive": 0.15,
        "rest": 0.18,
    },
    # Permeability-limited organs
    permeability_limited={
        "adipose": {"kp": 0.10, "ps": 5.0},
        "muscle": {"kp": 0.18, "ps": 10.0},
        "bone": {"kp": 0.12, "ps": 3.0},
        "skin": {"kp": 0.15, "ps": 5.0},
    },
    gut_clint_multiplier=0.1,
)
