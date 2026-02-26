"""Metformin — biguanide antidiabetic, renally cleared, very hydrophilic.

Literature references:
  - Tucker GT et al., Br J Clin Pharmacol, 1981 (renal clearance)
  - Sirtori CR et al., Clin Pharmacokinet, 1994 (bioavailability 50%)
  - Sambol NC et al., J Clin Pharmacol, 1996 (PK in healthy subjects)
  - Graham GG et al., Diabetologia, 2011 (mechanism and PK review)

Key PK properties:
  - Very hydrophilic: logP = -1.4, minimal protein binding (fup ~ 0.99)
  - Not hepatically metabolized: no CYP metabolism
  - Renally cleared via active tubular secretion (OCT2/MATE): CLr ~ 25-35 L/h
  - Oral bioavailability ~ 50% due to incomplete absorption (not first-pass)
  - Accumulates in gastrointestinal wall and some tissues
  - Low Vss ~ 63-276 L (volume-dependent on tissue uptake transporters)
"""

from omega_pbpk.drugs.drug import Drug

METFORMIN = Drug(
    name="Metformin",
    mw=129.16,
    logP=-1.4,
    pka=[12.4],
    drug_type="monoprotic_base",
    compound_type="base",
    partition_method="heuristic",
    fup=0.99,
    rbp=1.0,
    smiles="CN(C)C(=N)NC(=N)N",
    # No CYP metabolism
    clint={},
    fm={},
    # No hepatic metabolism
    clint_hepatic_L_per_h=0.0,
    clint_gut_L_per_h=0.0,
    # Renal clearance via active tubular secretion (OCT2/MATE)
    # CLr ~ 25-35 L/h on total plasma concentration basis (Tucker 1981)
    clr_L_per_h=25.0,
    # Absorption: BCS Class III — high solubility, low permeability
    peff=0.3,
    solubility_mg_mL=60.0,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — elevated in kidney (OCT2 uptake), gut (OCT1)
    kp={
        "lung": 2.0,
        "brain": 0.5,
        "heart": 1.5,
        "kidney": 5.0,
        "liver": 3.0,
        "spleen": 2.0,
        "gut_wall": 3.5,
        "pancreas": 2.5,
        "thymus": 1.5,
        "reproductive": 1.5,
        "rest": 1.0,
    },
    # Permeability-limited organs
    permeability_limited={
        "adipose": {"kp": 0.30, "ps": 2.0},
        "muscle": {"kp": 1.50, "ps": 5.0},
        "bone": {"kp": 0.50, "ps": 2.0},
        "skin": {"kp": 0.80, "ps": 3.0},
    },
    gut_clint_multiplier=0.0,
)
