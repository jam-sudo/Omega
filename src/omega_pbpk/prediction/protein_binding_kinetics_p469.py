"""Phase 469 — Plasma Protein Binding Kinetics.

Models on/off-rate binding kinetics of drugs to plasma proteins:
albumin, alpha-1-acid glycoprotein (AAG), and lipoprotein.
"""

from __future__ import annotations

from dataclasses import dataclass

# Protein concentrations in µM
_PROTEIN_CONC_UM: dict[str, float] = {
    "albumin": 600.0,
    "AAG": 1.0,
    "lipoprotein": 0.5,
}

_VALID_PROTEINS = frozenset(_PROTEIN_CONC_UM.keys())


@dataclass(frozen=True)
class ProteinBindingKineticsResult:
    """Result of plasma protein binding kinetics prediction."""

    drug_name: str
    protein: str
    kon_M_per_s: float
    koff_per_s: float
    kd_uM: float
    fu_equilibrium: float
    residence_time_s: float
    time_to_equilibrium_s: float
    binding_class: str
    displacement_risk: bool
    notes: str


def _classify_binding(kd_uM: float) -> str:
    """Classify binding strength based on Kd."""
    if kd_uM > 100.0:
        return "weak"
    if kd_uM >= 1.0:
        return "moderate"
    if kd_uM >= 0.01:
        return "strong"
    return "very_strong"


def predict_protein_binding_kinetics(
    drug_name: str,
    protein: str,
    kon_M_per_s: float,
    koff_per_s: float,
) -> ProteinBindingKineticsResult:
    """Predict plasma protein binding kinetics.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    protein : str
        Target protein: "albumin", "AAG", or "lipoprotein".
    kon_M_per_s : float
        Association rate constant (M^-1 s^-1).
    koff_per_s : float
        Dissociation rate constant (s^-1).

    Returns
    -------
    ProteinBindingKineticsResult
    """
    if kon_M_per_s <= 0.0:
        raise ValueError(f"kon_M_per_s must be > 0, got {kon_M_per_s}")
    if koff_per_s <= 0.0:
        raise ValueError(f"koff_per_s must be > 0, got {koff_per_s}")
    if protein not in _VALID_PROTEINS:
        raise ValueError(f"protein must be one of {sorted(_VALID_PROTEINS)}, got {protein!r}")

    # Kd in µM: (koff/kon) * 1e6  (koff in s^-1, kon in M^-1 s^-1 → Kd in M → ×1e6 → µM)
    kd_uM = (koff_per_s / kon_M_per_s) * 1e6

    # Protein concentration in µM
    p_conc_uM = _PROTEIN_CONC_UM[protein]

    # fu = 1 / (1 + [P]/Kd)  — [P] and Kd both in µM
    fu_equilibrium = 1.0 / (1.0 + p_conc_uM / kd_uM)

    residence_time_s = 1.0 / koff_per_s
    time_to_equilibrium_s = 5.0 / koff_per_s

    binding_class = _classify_binding(kd_uM)
    displacement_risk = fu_equilibrium < 0.1

    notes_parts = [
        f"Kd = {kd_uM:.4g} µM ({binding_class} binding).",
        f"fu = {fu_equilibrium:.4f} at [{protein}] = {p_conc_uM} µM.",
        f"Residence time = {residence_time_s:.4g} s; equilibrium in ~{time_to_equilibrium_s:.4g} s.",
    ]
    if displacement_risk:
        notes_parts.append("High binding (fu < 0.1): significant DDI displacement risk.")

    return ProteinBindingKineticsResult(
        drug_name=drug_name,
        protein=protein,
        kon_M_per_s=kon_M_per_s,
        koff_per_s=koff_per_s,
        kd_uM=kd_uM,
        fu_equilibrium=fu_equilibrium,
        residence_time_s=residence_time_s,
        time_to_equilibrium_s=time_to_equilibrium_s,
        binding_class=binding_class,
        displacement_risk=displacement_risk,
        notes=" ".join(notes_parts),
    )


def compare_proteins(
    drug_name: str,
    kon_M_per_s: float,
    koff_per_s: float,
) -> list[ProteinBindingKineticsResult]:
    """Compare binding kinetics across all three plasma proteins.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    kon_M_per_s : float
        Association rate constant (M^-1 s^-1).
    koff_per_s : float
        Dissociation rate constant (s^-1).

    Returns
    -------
    list[ProteinBindingKineticsResult]
        Results sorted by fu_equilibrium ascending (tightest binding first).
    """
    results = [
        predict_protein_binding_kinetics(drug_name, protein, kon_M_per_s, koff_per_s)
        for protein in sorted(_VALID_PROTEINS)
    ]
    return sorted(results, key=lambda r: r.fu_equilibrium)


__all__ = [
    "ProteinBindingKineticsResult",
    "predict_protein_binding_kinetics",
    "compare_proteins",
]


if __name__ == "__main__":  # pragma: no cover
    # Quick smoke test
    result = predict_protein_binding_kinetics(
        drug_name="TestDrug",
        protein="albumin",
        kon_M_per_s=1e6,
        koff_per_s=1.0,
    )
    print(result)
    print()
    for r in compare_proteins("TestDrug", 1e6, 1.0):
        print(f"  {r.protein}: fu={r.fu_equilibrium:.4f}, class={r.binding_class}")

    # Verify math: kd_uM = 1.0/1e6 * 1e6 = 1.0 µM
    assert abs(result.kd_uM - 1.0) < 1e-10, f"kd_uM mismatch: {result.kd_uM}"
    assert abs(result.residence_time_s - 1.0) < 1e-10
    assert abs(result.time_to_equilibrium_s - 5.0) < 1e-10
    print("Smoke test passed.")
