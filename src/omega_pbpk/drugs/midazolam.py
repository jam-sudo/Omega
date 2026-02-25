"""Midazolam — FDA PBPK probe drug (CYP3A4/5 substrate).

Literature references:
  - Thummel KE et al., J Pharmacol Exp Ther, 1996
  - Gorski JC et al., Clin Pharmacol Ther, 2003
  - Paine MF et al., Clin Pharmacol Ther, 2006 (Fg = 0.43-0.57)
  - Heizmann P et al., Br J Clin Pharmacol, 1983
"""

from omega_pbpk.drugs.drug import Drug

MIDAZOLAM = Drug(
    name="Midazolam",
    mw=325.77,
    logP=3.89,
    pka=[6.15],
    drug_type="monoprotic_base",
    fup=0.032,
    rbp=0.66,
    smiles="Clc1ccc2c(c1)C(=NCc3nccn3C)c1cc(F)ccc1N2",
    clint={
        "CYP3A4": 0.9,
        "CYP3A5": 0.3,
    },
    fm={
        "CYP3A4": 0.93,
        "CYP3A5": 0.04,
        "other": 0.03,
    },
    peff=5.37,
    solubility_mg_mL=0.024,
    particle_radius_um=25.0,
    particle_density=1.2,
    kp={
        "lung": 0.6,
        "brain": 6.61,
        "heart": 5.04,
        "kidney": 5.12,
        "liver": 5.87,
        "spleen": 3.8,
        "gut_wall": 4.2,
        "pancreas": 3.5,
        "thymus": 2.8,
        "reproductive": 3.0,
        "rest": 2.5,
    },
    permeability_limited={
        "adipose": {"kp": 30.0, "ps": 8.0},
        "muscle": {"kp": 5.5, "ps": 25.0},
        "bone": {"kp": 6.0, "ps": 5.0},
        "skin": {"kp": 6.0, "ps": 8.0},
    },
    gut_clint_multiplier=11.0,
)
