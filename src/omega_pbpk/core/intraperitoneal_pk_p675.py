"""Phase 675 — Intraperitoneal Drug Absorption PK Model."""

from dataclasses import dataclass, field


@dataclass
class IntraperitonealPKResult:
    """Result of intraperitoneal PK simulation."""

    drug_name: str
    dose_mg: float
    species: str
    times_h: list = field(default_factory=list)
    c_peritoneal_mg_L: list = field(default_factory=list)
    c_portal_mg_L: list = field(default_factory=list)
    c_systemic_mg_L: list = field(default_factory=list)
    cmax_systemic_mg_L: float = 0.0
    tmax_systemic_h: float = 0.0
    auc_systemic_mg_h_per_L: float = 0.0
    auc_peritoneal_mg_h_per_L: float = 0.0
    peritoneal_to_systemic_auc_ratio: float = 0.0
    first_pass_effect_pct: float = 0.0
    absorption_half_life_h: float = 0.0
    notes: str = ""


SPECIES_PARAMS = {
    "mouse": {"peritoneal_volume_L": 0.002, "hepatic_flow_L_h": 0.0048},
    "rat": {"peritoneal_volume_L": 0.01, "hepatic_flow_L_h": 0.048},
    "human": {"peritoneal_volume_L": 2.0, "hepatic_flow_L_h": 90.0},
}

VALID_SPECIES = set(SPECIES_PARAMS.keys())


def simulate_intraperitoneal_pk(
    drug_name: str,
    dose_mg: float,
    cl_sys_L_per_h: float,
    vd_sys_L: float,
    logP: float,
    mw_Da: float,
    species: str = "mouse",
    t_end_h: float = 12.0,
    dt_h: float = 0.05,
) -> IntraperitonealPKResult:
    """Simulate intraperitoneal drug absorption."""
    if dose_mg <= 0:
        raise ValueError("dose_mg must be > 0")
    if cl_sys_L_per_h <= 0:
        raise ValueError("cl_sys_L_per_h must be > 0")
    if vd_sys_L <= 0:
        raise ValueError("vd_sys_L must be > 0")
    if species not in VALID_SPECIES:
        raise ValueError(f"species must be one of {sorted(VALID_SPECIES)}")

    sp = SPECIES_PARAMS[species]
    peritoneal_volume_L = sp["peritoneal_volume_L"]
    hepatic_flow_L_h = sp["hepatic_flow_L_h"]

    ka_ip = max(0.1, min(3.0, 0.5 + logP * 0.2 - mw_Da / 1000.0))
    f_portal = 0.75
    f_lymph = 0.25

    E_h = hepatic_flow_L_h / (hepatic_flow_L_h + cl_sys_L_per_h * 0.5)
    E_h = min(0.99, E_h)
    first_pass_effect_pct = E_h * f_portal * 100.0

    V_portal = 0.5

    C_peri = dose_mg / peritoneal_volume_L
    C_portal = 0.0
    C_sys = 0.0

    times_h = []
    c_peritoneal = []
    c_portal = []
    c_systemic = []

    n_steps = int(t_end_h / dt_h)
    t = 0.0

    for _ in range(n_steps + 1):
        times_h.append(t)
        c_peritoneal.append(C_peri)
        c_portal.append(C_portal)
        c_systemic.append(C_sys)

        dC_peri = -ka_ip * C_peri
        dC_portal = (
            ka_ip * C_peri * peritoneal_volume_L * f_portal / V_portal
            - hepatic_flow_L_h * C_portal / V_portal
        )
        dC_sys = (
            hepatic_flow_L_h * C_portal * (1 - E_h) / vd_sys_L
            + ka_ip * C_peri * peritoneal_volume_L * f_lymph / vd_sys_L
            - (cl_sys_L_per_h / vd_sys_L) * C_sys
        )

        C_peri = max(0.0, C_peri + dC_peri * dt_h)
        C_portal = max(0.0, C_portal + dC_portal * dt_h)
        C_sys = max(0.0, C_sys + dC_sys * dt_h)

        t += dt_h

    cmax_systemic = max(c_systemic)
    tmax_systemic = times_h[c_systemic.index(cmax_systemic)]

    auc_systemic = sum(
        0.5 * (c_systemic[i] + c_systemic[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )
    auc_peritoneal = sum(
        0.5 * (c_peritoneal[i] + c_peritoneal[i - 1]) * (times_h[i] - times_h[i - 1])
        for i in range(1, len(times_h))
    )

    ratio = auc_peritoneal / auc_systemic if auc_systemic > 0 else 0.0
    absorption_half_life_h = 0.693 / ka_ip

    return IntraperitonealPKResult(
        drug_name=drug_name,
        dose_mg=dose_mg,
        species=species,
        times_h=times_h,
        c_peritoneal_mg_L=c_peritoneal,
        c_portal_mg_L=c_portal,
        c_systemic_mg_L=c_systemic,
        cmax_systemic_mg_L=cmax_systemic,
        tmax_systemic_h=tmax_systemic,
        auc_systemic_mg_h_per_L=auc_systemic,
        auc_peritoneal_mg_h_per_L=auc_peritoneal,
        peritoneal_to_systemic_auc_ratio=ratio,
        first_pass_effect_pct=first_pass_effect_pct,
        absorption_half_life_h=absorption_half_life_h,
        notes=f"IP PK simulation for {drug_name} in {species}",
    )
