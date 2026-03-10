#!/usr/bin/env python3
"""Generate benchmark C(t) profiles for 13 drugs using published PK parameters.

Uses the Bateman equation (one-compartment oral absorption model):
  C(t) = (F * Dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))

PK parameters sourced from:
- FDA-approved drug labels (DailyMed)
- Goodman & Gilman's Pharmacological Basis of Therapeutics
- Rowland & Tozer: Clinical Pharmacokinetics and Pharmacodynamics
- Individual published PK studies (cited per drug)
"""

import csv
import math
import os

# Standard timepoints matching existing benchmark format
TIMEPOINTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0]


def bateman(t, F, dose_mg, Vd_L, ka, ke):
    """One-compartment oral model (Bateman equation)."""
    if t == 0:
        return 0.0
    if abs(ka - ke) < 1e-10:
        return (F * dose_mg * ka * t / Vd_L) * math.exp(-ke * t)
    A = (F * dose_mg * ka) / (Vd_L * (ka - ke))
    return A * (math.exp(-ke * t) - math.exp(-ka * t))


def ka_from_tmax(tmax, ke):
    """Estimate ka from Tmax using Newton's method on Tmax = ln(ka/ke)/(ka-ke)."""
    # Start with ka = 3 * ke as initial guess
    ka = 3.0 * ke
    for _ in range(100):
        if ka <= ke:
            ka = ke * 2.0
        f = math.log(ka / ke) / (ka - ke) - tmax
        df = (1.0 / ka - math.log(ka / ke)) / (ka - ke) - math.log(ka / ke) / (ka - ke) ** 2
        # Simplified: use numerical derivative
        eps = ka * 1e-6
        ka2 = ka + eps
        f2 = math.log(ka2 / ke) / (ka2 - ke) - tmax
        df = (f2 - f) / eps
        if abs(df) < 1e-15:
            break
        ka_new = ka - f / df
        if ka_new <= ke:
            ka = (ka + ke) / 2.0  # bisect toward ke
        else:
            ka = ka_new
        if abs(f) < 1e-10:
            break
    return ka


# Drug PK parameters from published literature
# Format: (name, dose_mg, F, Vd_L, t_half_h, tmax_h, source)
DRUGS = [
    {
        "name": "ibuprofen",
        "dose_mg": 400.0,
        "F": 0.80,
        "Vd_L": 10.0,  # ~0.14 L/kg * 70kg
        "t_half_h": 2.0,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label; Davies 1998 Clin Pharmacokinet 34(2):101-54",
    },
    {
        "name": "diazepam",
        "dose_mg": 10.0,
        "F": 0.95,
        "Vd_L": 77.0,  # ~1.1 L/kg * 70kg
        "t_half_h": 43.0,
        "tmax_h": 1.0,
        "route": "oral",
        "source": "FDA label (Valium); Mandelli 1978 Clin Pharmacokinet 3(1):72-91",
    },
    {
        "name": "theophylline",
        "dose_mg": 300.0,
        "F": 0.96,
        "Vd_L": 35.0,  # ~0.5 L/kg * 70kg
        "t_half_h": 8.0,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label; Hendeles 1995 Pharmacotherapy 15(4):409-27",
    },
    {
        "name": "digoxin",
        "dose_mg": 0.5,
        "F": 0.70,
        "Vd_L": 490.0,  # ~7 L/kg * 70kg (large Vd)
        "t_half_h": 36.0,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label (Lanoxin); Reuning 1973 Am Heart J 85(5):663-70",
    },
    {
        "name": "acetaminophen",
        "dose_mg": 1000.0,
        "F": 0.85,
        "Vd_L": 63.0,  # ~0.9 L/kg * 70kg
        "t_half_h": 2.5,
        "tmax_h": 0.75,
        "route": "oral",
        "source": "FDA label (Tylenol); Forrest 1982 Clin Pharmacokinet 7(2):93-107",
    },
    {
        "name": "omeprazole",
        "dose_mg": 20.0,
        "F": 0.50,
        "Vd_L": 10.0,  # apparent Vd ~0.13-0.14 L/kg * 70kg
        "t_half_h": 1.0,
        "tmax_h": 0.75,
        "route": "oral",
        "source": "FDA label (Prilosec); Andersson 1990 Clin Pharmacokinet 19(1):44-64",
    },
    {
        "name": "amoxicillin",
        "dose_mg": 500.0,
        "F": 0.80,
        "Vd_L": 21.0,  # ~0.3 L/kg * 70kg
        "t_half_h": 1.5,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label; Sjovall 1986 J Antimicrob Chemother 18(6):763-9",
    },
    {
        "name": "atorvastatin",
        "dose_mg": 40.0,
        "F": 0.12,
        "Vd_L": 381.0,  # ~5.4 L/kg * 70kg
        "t_half_h": 14.0,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label (Lipitor); Lennernas 2003 Clin Pharmacokinet 42(13):1141-60",
    },
    {
        "name": "fluoxetine",
        "dose_mg": 20.0,
        "F": 0.72,
        "Vd_L": 2100.0,  # ~30 L/kg * 70kg (very large Vd)
        "t_half_h": 48.0,
        "tmax_h": 6.0,
        "route": "oral",
        "source": "FDA label (Prozac); Altamura 1994 Clin Pharmacokinet 26(3):201-14",
    },
    {
        "name": "carbamazepine",
        "dose_mg": 200.0,
        "F": 0.75,
        "Vd_L": 98.0,  # ~1.4 L/kg * 70kg
        "t_half_h": 36.0,
        "tmax_h": 6.0,
        "route": "oral",
        "source": "FDA label (Tegretol); Bertilsson 1986 Clin Pharmacokinet 11(3):177-98",
    },
    {
        "name": "phenytoin",
        "dose_mg": 300.0,
        "F": 0.90,
        "Vd_L": 45.0,  # ~0.64 L/kg * 70kg
        "t_half_h": 22.0,
        "tmax_h": 4.0,
        "route": "oral",
        "source": "FDA label (Dilantin); Browne 1983 Clin Pharmacokinet 8(3):196-209",
    },
    {
        "name": "verapamil",
        "dose_mg": 80.0,
        "F": 0.22,
        "Vd_L": 266.0,  # ~3.8 L/kg * 70kg
        "t_half_h": 6.0,
        "tmax_h": 1.5,
        "route": "oral",
        "source": "FDA label (Calan); Hamann 1984 Clin Pharmacokinet 9(1):26-41",
    },
    {
        "name": "nifedipine",
        "dose_mg": 10.0,
        "F": 0.50,
        "Vd_L": 56.0,  # ~0.8 L/kg * 70kg
        "t_half_h": 2.0,
        "tmax_h": 0.5,
        "route": "oral",
        "source": "FDA label (Procardia); Kleinbloesem 1984 Clin Pharmacokinet 9(5):397-414",
    },
]


def generate_drug_csv(drug, out_dir):
    """Generate C(t) CSV for a drug."""
    ke = math.log(2) / drug["t_half_h"]
    ka = ka_from_tmax(drug["tmax_h"], ke)

    rows = []
    for t in TIMEPOINTS:
        c = bateman(t, drug["F"], drug["dose_mg"], drug["Vd_L"], ka, ke)
        std = max(c * 0.20, 1e-7)  # 20% CV, minimum 1e-7
        rows.append((t, c, std))

    filename = f"{drug['name']}_oral_{int(drug['dose_mg'])}mg.csv"
    # Handle fractional doses
    if drug["dose_mg"] < 1:
        filename = f"{drug['name']}_oral_{drug['dose_mg']}mg.csv"

    filepath = os.path.join(out_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_h", "C_plasma_mg_per_L", "std_mg_per_L"])
        for t, c, std in rows:
            writer.writerow([t, c, std])

    peak_idx = max(range(len(rows)), key=lambda i: rows[i][1])
    print(
        f"  Generated {filename}: Cmax={max(r[1] for r in rows):.6g} mg/L at t={TIMEPOINTS[peak_idx]}h"
    )
    return filename


def generate_drug_config(drug, csv_filename, out_dir):
    """Generate YAML config for a drug."""
    filepath = os.path.join(out_dir, f"{drug['name']}.yaml")
    dose_val = drug["dose_mg"]

    with open(filepath, "w") as f:
        f.write(f"name: {drug['name']}\n")
        f.write(f"dataset: benchmarks/datasets/{csv_filename}\n")
        f.write(f"compound_file: compounds/{drug['name']}.yaml\n")
        f.write(f"dose_mg: {dose_val}\n")
        f.write(f"route: {drug['route']}\n")
        f.write("t_end_h: 24.0\n")
        f.write(f'source: "{drug["source"]}"\n')

    print(f"  Generated {drug['name']}.yaml config")


def main():
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    configs_dir = os.path.join(os.path.dirname(__file__), "configs")
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(configs_dir, exist_ok=True)

    print(f"Generating benchmark data for {len(DRUGS)} drugs...\n")
    for drug in DRUGS:
        print(f"Drug: {drug['name']} ({drug['dose_mg']}mg {drug['route']})")
        csv_fn = generate_drug_csv(drug, datasets_dir)
        generate_drug_config(drug, csv_fn, configs_dir)
        print()

    print(f"Done! Generated {len(DRUGS)} drug profiles.")
    print(f"Total benchmark drugs: 7 (existing) + {len(DRUGS)} (new) = {7 + len(DRUGS)}")


if __name__ == "__main__":
    main()
