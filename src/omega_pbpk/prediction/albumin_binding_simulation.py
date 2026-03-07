"""Drug-albumin binding equilibrium simulation — Phase 1096.

Scatchard 2-site albumin binding model (Sudlow Site I and Site II).
Iterative fixed-point solver — no numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Albumin MW = 66.5 kDa → 35 g/L / 66.5 kDa * 1000 µmol/mmol ≈ 526.3 µmol/L
_ALBUMIN_MW_g_per_mol = 66500.0  # g/mol


def _albumin_uM(albumin_g_per_L: float) -> float:
    """Convert albumin concentration from g/L to µmol/L."""
    return (albumin_g_per_L / _ALBUMIN_MW_g_per_mol) * 1e6


def _bound_site(D_free: float, Ka: float, n: float, alb_uM: float) -> float:
    """Bound drug (µmol/L) at a single Scatchard site.

    B = n * alb * D_free / (1/Ka + D_free)   (Kd = 1/Ka)
    """
    Kd = 1.0 / Ka if Ka > 0 else math.inf
    if Kd == math.inf:
        return 0.0
    return n * alb_uM * D_free / (Kd + D_free)


def _total_bound(
    D_free: float, Ka1: float, Ka2: float, n1: float, n2: float, alb_uM: float, primary_site: str
) -> float:
    """Total bound drug for a given D_free."""
    if primary_site == "site_I":
        return _bound_site(D_free, Ka1, n1, alb_uM)
    elif primary_site == "site_II":
        return _bound_site(D_free, Ka2, n2, alb_uM)
    else:
        return _bound_site(D_free, Ka1, n1, alb_uM) + _bound_site(D_free, Ka2, n2, alb_uM)


def _solve_free(
    D_total: float,
    Ka1: float,
    Ka2: float,
    n1: float,
    n2: float,
    alb_uM: float,
    primary_site: str,
    n_iter: int = 60,
) -> tuple[float, float, float]:
    """Solve for free drug concentration via bisection.

    We seek D_free such that: D_free + B_total(D_free) = D_total.
    Equivalently: f(D_free) = D_free + B_total(D_free) - D_total = 0.
    f is monotonically increasing in D_free, so bisection works.

    Returns (D_free, B_site1, B_site2).
    """
    if alb_uM <= 0.0:
        return D_total, 0.0, 0.0

    def f(Df: float) -> float:
        return Df + _total_bound(Df, Ka1, Ka2, n1, n2, alb_uM, primary_site) - D_total

    lo = 0.0
    hi = D_total

    # Extend hi until f(hi) >= 0 (should always hold: if D_free=D_total → B>=0)
    # f(0) = B(0) - D_total ≤ 0 (since B(0)=0); f(D_total) = D_total + B(D_total) - D_total = B(D_total) ≥ 0
    # So bisection bracket [0, D_total] is always valid.

    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < 1e-12 or (hi - lo) < 1e-14:
            lo = mid
            break
        if fmid < 0:
            lo = mid
        else:
            hi = mid

    D_free = lo
    D_free = max(0.0, min(D_free, D_total))

    # Final bound amounts at converged D_free
    if primary_site == "site_I":
        b1 = _bound_site(D_free, Ka1, n1, alb_uM)
        b2 = 0.0
    elif primary_site == "site_II":
        b1 = 0.0
        b2 = _bound_site(D_free, Ka2, n2, alb_uM)
    else:
        b1 = _bound_site(D_free, Ka1, n1, alb_uM)
        b2 = _bound_site(D_free, Ka2, n2, alb_uM)

    return D_free, b1, b2


@dataclass(frozen=True)
class AlbuminBindingResult:
    """Result of albumin binding prediction (frozen dataclass)."""

    drug_name: str
    drug_conc_uM: float
    site: str
    albumin_conc_g_per_L: float
    ka_site1_L_per_uM: float
    ka_site2_L_per_uM: float
    fu_predicted: float
    b_site1_uM: float
    b_site2_uM: float
    fraction_site1: float
    nonlinear_binding: bool
    concentration_dependent_fu: str
    notes: str


def predict_albumin_binding(
    drug_name: str,
    drug_conc_uM: float,
    ka_site1_L_per_uM: float = 2.0,
    ka_site2_L_per_uM: float = 0.5,
    albumin_conc_g_per_L: float = 35.0,
    primary_site: str = "both",
) -> AlbuminBindingResult:
    """Predict albumin binding using 2-site Scatchard model.

    Parameters
    ----------
    drug_name : str
        Name of the drug.
    drug_conc_uM : float
        Total drug concentration (µmol/L).
    ka_site1_L_per_uM : float
        Association constant for Sudlow Site I (L/µmol). Default 2.0.
    ka_site2_L_per_uM : float
        Association constant for Sudlow Site II (L/µmol). Default 0.5.
    albumin_conc_g_per_L : float
        Albumin concentration (g/L). Default 35 (normal plasma).
    primary_site : str
        Which sites to include: "site_I", "site_II", or "both".

    Returns
    -------
    AlbuminBindingResult
    """
    # Validation
    if drug_conc_uM <= 0:
        raise ValueError(f"drug_conc_uM must be > 0, got {drug_conc_uM}")
    if albumin_conc_g_per_L < 0:
        raise ValueError(f"albumin_conc_g_per_L must be >= 0, got {albumin_conc_g_per_L}")
    if ka_site1_L_per_uM <= 0:
        raise ValueError(f"ka_site1_L_per_uM must be > 0, got {ka_site1_L_per_uM}")
    if ka_site2_L_per_uM <= 0:
        raise ValueError(f"ka_site2_L_per_uM must be > 0, got {ka_site2_L_per_uM}")
    if primary_site not in {"site_I", "site_II", "both"}:
        raise ValueError(
            f"primary_site must be 'site_I', 'site_II', or 'both', got {primary_site!r}"
        )

    alb_uM = _albumin_uM(albumin_conc_g_per_L)

    D_free, b1, b2 = _solve_free(
        drug_conc_uM,
        ka_site1_L_per_uM,
        ka_site2_L_per_uM,
        1.0,
        1.0,
        alb_uM,
        primary_site,
    )

    fu = D_free / drug_conc_uM if drug_conc_uM > 0 else 1.0
    fu = min(1.0, max(0.0, fu))

    total_bound = b1 + b2
    fraction_s1 = b1 / total_bound if total_bound > 0 else 0.0

    # Check concentration-dependence: compare fu at 1x vs 10x
    D_free_10x, b1_10x, b2_10x = _solve_free(
        drug_conc_uM * 10,
        ka_site1_L_per_uM,
        ka_site2_L_per_uM,
        1.0,
        1.0,
        alb_uM,
        primary_site,
    )
    fu_10x = (D_free_10x / (drug_conc_uM * 10)) if drug_conc_uM > 0 else 1.0
    fu_10x = min(1.0, max(0.0, fu_10x))

    nonlinear = abs(fu_10x - fu) > 0.10 * fu if fu > 0 else False
    conc_dep_str = "nonlinear" if nonlinear else "linear"

    notes = (
        f"Albumin={albumin_conc_g_per_L} g/L ({alb_uM:.1f} µM). "
        f"Site I Ka={ka_site1_L_per_uM} L/µmol, Site II Ka={ka_site2_L_per_uM} L/µmol. "
        f"fu={fu:.4f}. Binding: {conc_dep_str}."
    )

    return AlbuminBindingResult(
        drug_name=drug_name,
        drug_conc_uM=drug_conc_uM,
        site=primary_site,
        albumin_conc_g_per_L=albumin_conc_g_per_L,
        ka_site1_L_per_uM=ka_site1_L_per_uM,
        ka_site2_L_per_uM=ka_site2_L_per_uM,
        fu_predicted=fu,
        b_site1_uM=b1,
        b_site2_uM=b2,
        fraction_site1=fraction_s1,
        nonlinear_binding=nonlinear,
        concentration_dependent_fu=conc_dep_str,
        notes=notes,
    )


def fu_concentration_curve(
    drug_name: str,
    conc_range_uM: list[float],
    ka_site1_L_per_uM: float = 2.0,
    ka_site2_L_per_uM: float = 0.5,
    albumin_conc_g_per_L: float = 35.0,
    primary_site: str = "both",
) -> list[AlbuminBindingResult]:
    """Compute fu across a range of drug concentrations.

    Parameters
    ----------
    drug_name : str
        Drug name.
    conc_range_uM : list[float]
        List of total drug concentrations (µmol/L).
    **kwargs : dict
        Passed to predict_albumin_binding.

    Returns
    -------
    list[AlbuminBindingResult]
        Results sorted by drug_conc_uM ascending.
    """
    results = [
        predict_albumin_binding(
            drug_name=drug_name,
            drug_conc_uM=c,
            ka_site1_L_per_uM=ka_site1_L_per_uM,
            ka_site2_L_per_uM=ka_site2_L_per_uM,
            albumin_conc_g_per_L=albumin_conc_g_per_L,
            primary_site=primary_site,
        )
        for c in conc_range_uM
    ]
    results.sort(key=lambda r: r.drug_conc_uM)
    return results


def compare_affinities(
    drug_name: str,
    drug_conc_uM: float = 1.0,
    ka1_list: list[float] | None = None,
    ka_site2_L_per_uM: float = 0.5,
    albumin_conc_g_per_L: float = 35.0,
    primary_site: str = "both",
) -> list[AlbuminBindingResult]:
    """Compare albumin binding across different Site I affinities.

    Parameters
    ----------
    drug_name : str
        Drug name prefix (Ka will be appended).
    drug_conc_uM : float
        Drug concentration (µmol/L).
    ka1_list : list[float]
        List of Ka values for Site I (L/µmol) to compare.
    ka_site2_L_per_uM : float
        Ka for Site II (fixed).
    albumin_conc_g_per_L : float
        Albumin concentration (g/L).
    primary_site : str
        Site selectivity.

    Returns
    -------
    list[AlbuminBindingResult]
        Sorted by fu_predicted ascending (tighter binding = lower fu).
    """
    if ka1_list is None:
        ka1_list = [0.1, 0.5, 2.0, 5.0, 10.0]

    results = [
        predict_albumin_binding(
            drug_name=f"{drug_name}_Ka1_{ka:.2f}",
            drug_conc_uM=drug_conc_uM,
            ka_site1_L_per_uM=ka,
            ka_site2_L_per_uM=ka_site2_L_per_uM,
            albumin_conc_g_per_L=albumin_conc_g_per_L,
            primary_site=primary_site,
        )
        for ka in ka1_list
    ]
    results.sort(key=lambda r: r.fu_predicted)
    return results
