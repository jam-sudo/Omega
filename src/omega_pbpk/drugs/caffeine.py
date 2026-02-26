"""Caffeine — methylxanthine, CYP1A2 substrate, low protein binding, neutral.

Literature references:
  - Arnaud MJ, Food Chem Toxicol, 1993 (absorption and metabolism)
  - Busto U et al., Clin Pharmacol Ther, 1989 (PK parameters)
  - Denaro CP et al., Eur J Clin Pharmacol, 1990 (CYP1A2 substrate)
  - Rodopoulos N et al., Scand J Clin Lab Invest, 1996 (urinary metabolites)

Key PK properties:
  - Near-complete oral bioavailability: F ~ 99%
  - Low protein binding: fup = 0.65
  - Moderate volume of distribution: Vss ~ 0.5-0.7 L/kg
  - Primary metabolic route: CYP1A2 (paraxanthine, theobromine, theophylline)
  - Minor renal clearance of unchanged drug
  - t1/2 ~ 3-5 h in healthy adults (longer in neonates/hepatic impairment)
"""

from omega_pbpk.drugs.drug import Drug

CAFFEINE = Drug(
    name="Caffeine",
    mw=194.19,
    logP=-0.07,
    pka=[0.6],
    drug_type="neutral",
    compound_type="neutral",
    partition_method="heuristic",
    fup=0.65,
    rbp=1.0,
    smiles="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    # In vitro CLint (reference values)
    clint={
        "CYP1A2": 0.45,
        "CYP2E1": 0.02,
    },
    fm={
        "CYP1A2": 0.95,
        "CYP2E1": 0.03,
        "other": 0.02,
    },
    # Calibrated in vivo CLint (L/h) — retrograde from clinical PK
    # CL_plasma ~ 1.5-2.0 L/h, t1/2 ~ 3-5 h (Busto 1989, Denaro 1990)
    clint_hepatic_L_per_h=40.0,
    clint_gut_L_per_h=0.0,
    clr_L_per_h=0.08,
    # Absorption: BCS Class I — very high permeability and solubility
    peff=7.0,
    solubility_mg_mL=21.6,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — near-unity reflecting low protein binding and neutral drug
    kp={
        "lung": 0.82,
        "brain": 0.98,
        "heart": 0.83,
        "kidney": 1.00,
        "liver": 0.95,
        "spleen": 0.87,
        "gut_wall": 0.92,
        "pancreas": 0.85,
        "thymus": 0.80,
        "reproductive": 0.82,
        "rest": 0.80,
    },
    # Permeability-limited organs
    permeability_limited={
        "adipose": {"kp": 0.30, "ps": 20.0},
        "muscle": {"kp": 0.82, "ps": 40.0},
        "bone": {"kp": 0.60, "ps": 10.0},
        "skin": {"kp": 0.75, "ps": 15.0},
    },
    gut_clint_multiplier=0.1,
)
