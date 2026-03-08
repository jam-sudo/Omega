"""Phase 588 - Drug Half-Life Prediction from Structure.

Predicts drug half-life from structural features (logP, MW, PSA, HBD,
rotatable bonds) using simplified empirical correlations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StructuralHalflifeResult:
    """Result of structural half-life prediction."""

    drug_name: str
    logP: float
    mw_Da: float
    psa_A2: float
    hbd: int
    rot_bonds: int
    predicted_cl_L_per_h: float
    predicted_vd_L: float
    predicted_thalf_h: float
    thalf_class: str
    predicted_dosing_frequency: str
    high_clearance_alert: bool
    renal_or_hepatic_primary: str
    notes: str


def predict_halflife_from_structure(
    drug_name: str,
    logP: float,
    mw_Da: float,
    psa_A2: float,
    hbd: int,
    rot_bonds: int = 5,
) -> StructuralHalflifeResult:
    """Predict drug half-life from structural descriptors.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    logP : float
        Partition coefficient (log P).
    mw_Da : float
        Molecular weight in Daltons.
    psa_A2 : float
        Polar surface area in A^2.
    hbd : int
        Number of hydrogen bond donors.
    rot_bonds : int
        Number of rotatable bonds (default 5).

    Returns
    -------
    StructuralHalflifeResult
    """
    if mw_Da <= 0:
        raise ValueError("mw_Da must be > 0")
    if hbd < 0:
        raise ValueError("hbd must be >= 0")
    if psa_A2 < 0:
        raise ValueError("psa_A2 must be >= 0")

    # CL prediction
    cl_base = 15.0
    cl_lipo = cl_base * max(0.5, (logP + 2) / 4)
    cl_hydro = cl_base * max(0.5, 2.0 - logP * 0.3)
    predicted_cl = max(1.0, min(100.0, (cl_lipo + cl_hydro) / 2))

    # Vd prediction
    vd_base = 50.0
    vd = vd_base * (1 + 0.3 * max(0, logP - 1)) * (mw_Da / 300)
    predicted_vd = max(5.0, min(500.0, vd))

    # Half-life
    predicted_thalf = 0.693 * predicted_vd / predicted_cl

    # Classification
    if predicted_thalf < 1.0:
        thalf_class = "ultra_short"
    elif predicted_thalf < 6.0:
        thalf_class = "short"
    elif predicted_thalf <= 24.0:
        thalf_class = "intermediate"
    else:
        thalf_class = "long"

    # Dosing frequency
    if predicted_thalf < 6.0:
        dosing_freq = "QID"
    elif predicted_thalf < 8.0:
        dosing_freq = "TID"
    elif predicted_thalf < 16.0:
        dosing_freq = "BID"
    else:
        dosing_freq = "QD"

    high_clearance_alert = predicted_cl > 60.0
    renal_or_hepatic = "hepatic" if logP > 1 else "renal"

    notes = (
        f"Structural half-life prediction for {drug_name}. "
        f"Predicted CL={predicted_cl:.2f} L/h, Vd={predicted_vd:.1f} L, "
        f"t1/2={predicted_thalf:.2f} h ({thalf_class}). "
        f"Primary elimination: {renal_or_hepatic}."
    )

    return StructuralHalflifeResult(
        drug_name=drug_name,
        logP=logP,
        mw_Da=mw_Da,
        psa_A2=psa_A2,
        hbd=hbd,
        rot_bonds=rot_bonds,
        predicted_cl_L_per_h=predicted_cl,
        predicted_vd_L=predicted_vd,
        predicted_thalf_h=predicted_thalf,
        thalf_class=thalf_class,
        predicted_dosing_frequency=dosing_freq,
        high_clearance_alert=high_clearance_alert,
        renal_or_hepatic_primary=renal_or_hepatic,
        notes=notes,
    )


def screen_halflife(compounds: list) -> list:
    """Screen multiple compounds and sort by predicted half-life descending.

    Parameters
    ----------
    compounds : list of dict
        Each dict has keys: drug_name, logP, mw_Da, psa_A2, hbd,
        and optionally rot_bonds.

    Returns
    -------
    list of StructuralHalflifeResult sorted by predicted_thalf_h descending.
    """
    results = []
    for comp in compounds:
        result = predict_halflife_from_structure(
            drug_name=comp["drug_name"],
            logP=comp["logP"],
            mw_Da=comp["mw_Da"],
            psa_A2=comp["psa_A2"],
            hbd=comp["hbd"],
            rot_bonds=comp.get("rot_bonds", 5),
        )
        results.append(result)

    results.sort(key=lambda r: r.predicted_thalf_h, reverse=True)
    return results
