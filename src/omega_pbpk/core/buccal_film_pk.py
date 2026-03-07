"""Phase 1115 — Buccal film PK model (controlled-release mucosal drug delivery)."""

from dataclasses import dataclass


@dataclass
class BuccalFilmPKResult:
    """Result of buccal film PK simulation."""

    drug_name: str
    dose_mg: float
    f_buccal: float
    times_h: list
    c_plasma_mg_L: list
    a_film_mg: list
    a_mucosa_mg: list
    cmax_mg_L: float
    tmax_h: float
    auc_mg_h_per_L: float
    bioavailability_pct: float
    film_dissolved_pct: float
    notes: str


def simulate_buccal_film_pk(
    drug_name: str,
    dose_mg: float,
    k_dissolve_per_h: float = 0.3,
    f_buccal: float = 0.6,
    k_abs_per_h: float = 0.8,
    cl_L_per_h: float = 5.0,
    vd_L: float = 30.0,
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> BuccalFilmPKResult:
    """Simulate PK of a drug delivered via buccal film.

    Compartments:
      A_film   (mg) — drug remaining in buccal film (first-order dissolution)
      A_mucosa (mg) — drug in buccal mucosa tissue (delayed absorption)
      C_plasma (mg/L) — systemic plasma concentration

    ODEs (Forward Euler):
      dA_film/dt   = -k_dissolve * A_film
      dA_mucosa/dt = k_dissolve * f_buccal * A_film - k_abs * A_mucosa
      dC_plasma/dt = k_abs * A_mucosa / Vd - CL/Vd * C_plasma
                     + k_dissolve * (1 - f_buccal) * A_film / Vd

    Args:
        drug_name: Name of the drug.
        dose_mg: Total drug dose in buccal film (mg).
        k_dissolve_per_h: Film dissolution rate constant (1/h).
        f_buccal: Fraction absorbed buccally; (1-f_buccal) is swallowed.
        k_abs_per_h: Buccal mucosal absorption rate constant (1/h).
        cl_L_per_h: Systemic clearance (L/h).
        vd_L: Volume of distribution (L).
        t_end_h: Simulation end time (h).
        dt_h: Forward Euler time step (h).

    Returns:
        BuccalFilmPKResult with time-course and summary metrics.

    Raises:
        ValueError: For invalid parameter values.
    """
    # --- Validation ---
    if dose_mg <= 0:
        raise ValueError(f"dose_mg must be > 0, got {dose_mg}")
    if cl_L_per_h <= 0:
        raise ValueError(f"cl_L_per_h must be > 0, got {cl_L_per_h}")
    if vd_L <= 0:
        raise ValueError(f"vd_L must be > 0, got {vd_L}")
    if not (0 < f_buccal <= 1):
        raise ValueError(f"f_buccal must be in (0, 1], got {f_buccal}")
    if k_dissolve_per_h <= 0:
        raise ValueError(f"k_dissolve_per_h must be > 0, got {k_dissolve_per_h}")
    if k_abs_per_h <= 0:
        raise ValueError(f"k_abs_per_h must be > 0, got {k_abs_per_h}")
    if t_end_h <= 0:
        raise ValueError(f"t_end_h must be > 0, got {t_end_h}")
    if dt_h <= 0:
        raise ValueError(f"dt_h must be > 0, got {dt_h}")

    # --- ODE integration (Forward Euler) ---
    n_steps = int(t_end_h / dt_h) + 1
    times: list = []
    c_plasma_list: list = []
    a_film_list: list = []
    a_mucosa_list: list = []

    a_film = dose_mg
    a_mucosa = 0.0
    c_plasma = 0.0
    t = 0.0

    ke = cl_L_per_h / vd_L  # elimination rate constant

    for _ in range(n_steps):
        times.append(t)
        c_plasma_list.append(c_plasma)
        a_film_list.append(a_film)
        a_mucosa_list.append(a_mucosa)

        # Derivatives
        d_film = -k_dissolve_per_h * a_film
        d_mucosa = k_dissolve_per_h * f_buccal * a_film - k_abs_per_h * a_mucosa
        d_plasma = (
            k_abs_per_h * a_mucosa / vd_L
            - ke * c_plasma
            + k_dissolve_per_h * (1.0 - f_buccal) * a_film / vd_L
        )

        # Forward Euler update
        a_film += d_film * dt_h
        a_mucosa += d_mucosa * dt_h
        c_plasma += d_plasma * dt_h

        # Clamp negatives due to numerical error
        if a_film < 0.0:
            a_film = 0.0
        if a_mucosa < 0.0:
            a_mucosa = 0.0
        if c_plasma < 0.0:
            c_plasma = 0.0

        t += dt_h

    # --- Summary metrics ---
    cmax = max(c_plasma_list)
    tmax = times[c_plasma_list.index(cmax)]

    # Manual trapezoidal AUC
    auc = 0.0
    for i in range(1, len(times)):
        auc += 0.5 * (c_plasma_list[i - 1] + c_plasma_list[i]) * (times[i] - times[i - 1])

    # Theoretical IV AUC = dose / CL (all drug absorbed systemically)
    auc_iv_reference = dose_mg / cl_L_per_h
    bioavailability_pct = (auc / auc_iv_reference) * 100.0 if auc_iv_reference > 0 else 0.0

    # Percent of film dissolved
    film_dissolved_pct = (1.0 - a_film_list[-1] / dose_mg) * 100.0
    film_dissolved_pct = max(0.0, min(100.0, film_dissolved_pct))

    notes = (
        f"Buccal film PK simulation for {drug_name}. "
        f"f_buccal={f_buccal:.2f}, k_dissolve={k_dissolve_per_h:.3f}/h, "
        f"k_abs={k_abs_per_h:.3f}/h. "
        f"Swallowed fraction: {(1 - f_buccal) * 100:.1f}%."
    )

    return BuccalFilmPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        f_buccal=f_buccal,
        times_h=times,
        c_plasma_mg_L=c_plasma_list,
        a_film_mg=a_film_list,
        a_mucosa_mg=a_mucosa_list,
        cmax_mg_L=cmax,
        tmax_h=tmax,
        auc_mg_h_per_L=auc,
        bioavailability_pct=bioavailability_pct,
        film_dissolved_pct=film_dissolved_pct,
        notes=notes,
    )
