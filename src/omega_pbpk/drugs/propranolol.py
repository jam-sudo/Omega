"""Propranolol — beta-blocker, CYP2D6/1A2 substrate, basic drug, high ER.

Literature references:
  - Evans GH et al., J Pharmacol Exp Ther, 1973 (first-pass effect)
  - Walle T et al., J Pharmacol Exp Ther, 1985 (CYP2D6 metabolism)
  - Olanoff LS et al., Clin Pharmacol Ther, 1984 (PK parameters)
  - Masubuchi Y et al., Drug Metab Dispos, 1994 (CYP1A2 contribution)

Key PK properties:
  - High extraction ratio: F ~ 25-30% (extensive first-pass)
  - Moderate plasma protein binding: fup = 0.13
  - High lipophilicity: logP = 3.48, high Vss ~ 250-350 L
  - Primary route: CYP2D6 (ring hydroxylation); CYP1A2 (N-dealkylation)
  - Renal clearance minor (CLr ~ 0.5 L/h)
"""

from omega_pbpk.drugs.drug import Drug

PROPRANOLOL = Drug(
    name="Propranolol",
    mw=259.34,
    logP=3.48,
    pka=[9.5],
    drug_type="monoprotic_base",
    compound_type="base",
    partition_method="heuristic",
    fup=0.13,
    rbp=0.81,
    smiles="CC(C)NCC(O)COc1cccc2ccccc12",
    # In vitro CLint (reference values)
    clint={
        "CYP2D6": 0.55,
        "CYP1A2": 0.20,
    },
    fm={
        "CYP2D6": 0.70,
        "CYP1A2": 0.25,
        "other": 0.05,
    },
    # Calibrated in vivo CLint (L/h) — retrograde from clinical PK
    # High ER drug: CL_plasma ~ 60-80 L/h (literature)
    # Hepatic extraction ratio EH ~ 0.80-0.90 -> F_h ~ 0.10-0.20
    clint_hepatic_L_per_h=850.0,
    clint_gut_L_per_h=0.0,
    clr_L_per_h=0.5,
    # Absorption: moderate oral absorption limited by first-pass
    peff=4.2,
    solubility_mg_mL=0.11,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — high distribution reflecting basic lipophilic drug
    kp={
        "lung": 8.0,
        "brain": 10.5,
        "heart": 12.0,
        "kidney": 8.5,
        "liver": 15.0,
        "spleen": 7.0,
        "gut_wall": 9.0,
        "pancreas": 6.5,
        "thymus": 5.5,
        "reproductive": 6.0,
        "rest": 5.0,
    },
    # Permeability-limited organs — Kp calibrated for Vss ~ 300 L
    permeability_limited={
        "adipose": {"kp": 6.0, "ps": 15.0},
        "muscle": {"kp": 5.5, "ps": 30.0},
        "bone": {"kp": 3.0, "ps": 8.0},
        "skin": {"kp": 4.5, "ps": 12.0},
    },
    gut_clint_multiplier=0.5,
)
