"""Phase 579 - Hepatic Sinusoid Drug Transport Model.

Models OATP/MRP transporter-mediated drug uptake and efflux in hepatic sinusoids.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class HepaticSinusoidTransportResult:
    """Result of hepatic sinusoid transporter-mediated transport prediction."""

    drug_name: str
    oatp_activity: float
    mrp2_activity: float
    cl_uptake_L_per_h: float
    cl_efflux_L_per_h: float
    cl_net_L_per_h: float
    extraction_ratio_net: float
    f_oral_transporter: float
    ddi_vulnerability: str
    pgx_variability: str
    notes: str


Q_PORTAL = 72.0  # L/h


def predict_sinusoid_transport(
    drug_name: str,
    cl_passive_L_per_h: float,
    oatp_km_mg_L: float,
    oatp_vmax_L_per_h: float,
    mrp2_km_mg_L: float,
    mrp2_vmax_L_per_h: float,
    fup: float = 0.1,
    oatp_activity: float = 1.0,
    mrp2_activity: float = 1.0,
) -> HepaticSinusoidTransportResult:
    """Predict sinusoid transporter-mediated hepatic drug transport."""
    if not (0.0 < fup <= 1.0):
        raise ValueError("fup must be in (0, 1]")
    if oatp_vmax_L_per_h <= 0:
        raise ValueError("oatp_vmax_L_per_h must be > 0")
    if oatp_km_mg_L <= 0:
        raise ValueError("oatp_km_mg_L must be > 0")
    if mrp2_vmax_L_per_h <= 0:
        raise ValueError("mrp2_vmax_L_per_h must be > 0")
    if mrp2_km_mg_L <= 0:
        raise ValueError("mrp2_km_mg_L must be > 0")

    cl_passive_uptake = cl_passive_L_per_h * fup
    cl_oatp = oatp_activity * oatp_vmax_L_per_h / (1.0 + oatp_km_mg_L / 1.0)
    cl_total_uptake = cl_passive_uptake + cl_oatp

    cl_efflux = mrp2_activity * mrp2_vmax_L_per_h / (1.0 + mrp2_km_mg_L / 0.1)

    cl_net = max(0.0, cl_total_uptake - cl_efflux)

    e_h_raw = cl_net / (cl_net + Q_PORTAL) if (cl_net + Q_PORTAL) > 0 else 0.0
    e_h = max(0.0, min(1.0, e_h_raw))

    f_oral = 1.0 - e_h

    # DDI vulnerability
    if cl_oatp > cl_passive_uptake:
        ddi_vulnerability = "high"
    elif cl_oatp > 0.5 * cl_passive_uptake:
        ddi_vulnerability = "moderate"
    else:
        ddi_vulnerability = "low"

    # PGx variability
    oatp_fraction = cl_oatp / (cl_total_uptake + 0.001)
    pgx_variability = "significant" if oatp_fraction > 0.3 else "minimal"

    notes_parts = []
    notes_parts.append(f"Portal flow {Q_PORTAL} L/h")
    notes_parts.append(f"OATP CL {cl_oatp:.4f} L/h")
    notes_parts.append(f"Passive uptake CL {cl_passive_uptake:.4f} L/h")
    notes = "; ".join(notes_parts)

    return HepaticSinusoidTransportResult(
        drug_name=drug_name,
        oatp_activity=oatp_activity,
        mrp2_activity=mrp2_activity,
        cl_uptake_L_per_h=cl_total_uptake,
        cl_efflux_L_per_h=cl_efflux,
        cl_net_L_per_h=cl_net,
        extraction_ratio_net=e_h,
        f_oral_transporter=f_oral,
        ddi_vulnerability=ddi_vulnerability,
        pgx_variability=pgx_variability,
        notes=notes,
    )


def compare_transporter_activities(
    drug_name: str,
    cl_passive: float,
    oatp_km: float,
    oatp_vmax: float,
    mrp2_km: float,
    mrp2_vmax: float,
    scenarios: list,
) -> list:
    """Compare transporter activity scenarios, sorted by f_oral_transporter descending."""
    results = []
    for sc in scenarios:
        r = predict_sinusoid_transport(
            drug_name=drug_name,
            cl_passive_L_per_h=cl_passive,
            oatp_km_mg_L=oatp_km,
            oatp_vmax_L_per_h=oatp_vmax,
            mrp2_km_mg_L=mrp2_km,
            mrp2_vmax_L_per_h=mrp2_vmax,
            fup=sc.get("fup", 0.1),
            oatp_activity=sc.get("oatp_activity", 1.0),
            mrp2_activity=sc.get("mrp2_activity", 1.0),
        )
        results.append(r)
    results.sort(key=lambda x: x.f_oral_transporter, reverse=True)
    return results
