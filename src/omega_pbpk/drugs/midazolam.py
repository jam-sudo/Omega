"""Midazolam — FDA PBPK probe drug (CYP3A4/5 substrate).

Literature references:
  - Thummel KE et al., J Pharmacol Exp Ther, 1996
  - Gorski JC et al., Clin Pharmacol Ther, 2003
  - Paine MF et al., Clin Pharmacol Ther, 2006 (Fg = 0.43-0.57)
  - Heizmann P et al., Br J Clin Pharmacol, 1983
  - Gertz M et al., Drug Metab Dispos, 2010 (retrograde CLint)
  - Guest EJ et al., Eur J Pharm Sci, 2011 (Simcyp midazolam)

Calibrated in vivo CLint values (retrograde from observed clinical PK):
  CLint_hepatic = 200 L/h  (gives CLh ~ 6.3 L/h, t1/2 ~ 4.6 h IV)
  CLint_gut     = 2000 L/h (gives Fg ~ 0.29 at steady state)

Calibration target: 7.5 mg oral, healthy adults
  Cmax ~ 50 ng/mL, AUC ~ 90 ng*h/mL, t1/2 ~ 2.25 h, F ~ 36%
  Achieved AAFE(Cmax, AUC, t1/2) = 1.35

Kp values for permeability-limited organs calibrated to Vss ~ 90 L
(literature 77-119 L for midazolam).
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
    # In vitro CLint (kept for reference / IVIVE pathway)
    clint={
        "CYP3A4": 0.9,
        "CYP3A5": 0.3,
    },
    fm={
        "CYP3A4": 0.93,
        "CYP3A5": 0.04,
        "other": 0.03,
    },
    # Calibrated in vivo CLint (L/h) — retrograde from clinical PK
    clint_hepatic_L_per_h=200.0,
    clint_gut_L_per_h=2000.0,
    # Absorption (BCS Class I — high permeability, low solubility)
    peff=5.37,
    solubility_mg_mL=0.024,
    particle_radius_um=25.0,
    particle_density=1.2,
    # Tissue:plasma Kp — perfusion-limited organs
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
    # Permeability-limited organs — Kp calibrated for Vss ~ 90 L
    permeability_limited={
        "adipose": {"kp": 0.5, "ps": 8.0},
        "muscle": {"kp": 1.0, "ps": 25.0},
        "bone": {"kp": 1.5, "ps": 5.0},
        "skin": {"kp": 2.0, "ps": 8.0},
    },
    gut_clint_multiplier=11.0,
)
