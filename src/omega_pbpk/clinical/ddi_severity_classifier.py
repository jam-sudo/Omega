"""
Drug-Drug Interaction (DDI) Severity Classifier.

Classifies DDI severity based on AUCR thresholds and clinical endpoints,
following FDA 2020 DDI guidance.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------


def classify_ddi_severity(
    aucr: float,
    mechanism: str,
    narrow_therapeutic_index: bool = False,
) -> dict:
    """
    Classify DDI severity from an AUC ratio.

    Parameters
    ----------
    aucr:
        AUC ratio (victim AUC with perpetrator / victim AUC without perpetrator).
        Values > 1 indicate inhibition; values < 1 indicate induction.
    mechanism:
        One of 'inhibition', 'induction', or 'transporter'.
    narrow_therapeutic_index:
        If True, apply tighter thresholds (NTI drugs).

    Returns
    -------
    dict with keys:
        severity, mechanism, aucr, recommendation, label_warning_needed
    """
    mechanism = mechanism.lower().strip()
    if mechanism not in ("inhibition", "induction", "transporter"):
        raise ValueError(
            f"mechanism must be 'inhibition', 'induction', or 'transporter'; got '{mechanism}'"
        )
    if aucr <= 0:
        raise ValueError("aucr must be > 0")

    if mechanism == "inhibition" or mechanism == "transporter":
        severity = _classify_inhibition(aucr, narrow_therapeutic_index)
    else:  # induction
        severity = _classify_induction(aucr, narrow_therapeutic_index)

    recommendation = _build_recommendation(severity, mechanism, narrow_therapeutic_index)
    label_warning_needed = severity in ("strong", "moderate")

    return {
        "severity": severity,
        "mechanism": mechanism,
        "aucr": aucr,
        "recommendation": recommendation,
        "label_warning_needed": label_warning_needed,
    }


def _classify_inhibition(aucr: float, nti: bool) -> str:
    """Classify inhibition severity from AUCR."""
    if nti:
        # Tighter thresholds for NTI drugs
        if aucr >= 2.0:
            return "strong"
        elif aucr >= 1.5:
            return "moderate"
        elif aucr >= 1.1:
            return "weak"
        else:
            return "minimal"
    else:
        if aucr >= 5.0:
            return "strong"
        elif aucr >= 2.0:
            return "moderate"
        elif aucr >= 1.25:
            return "weak"
        else:
            return "minimal"


def _classify_induction(aucr: float, nti: bool) -> str:
    """Classify induction severity from AUCR (values < 1 indicate induction)."""
    if nti:
        # Tighter thresholds for NTI drugs
        if aucr <= 0.5:
            return "strong"
        elif aucr <= 0.7:
            return "moderate"
        elif aucr <= 0.9:
            return "weak"
        else:
            return "minimal"
    else:
        if aucr <= 0.2:
            return "strong"
        elif aucr <= 0.5:
            return "moderate"
        elif aucr <= 0.8:
            return "weak"
        else:
            return "minimal"


def _build_recommendation(severity: str, mechanism: str, nti: bool) -> str:
    """Build a clinical recommendation string."""
    nti_note = " (NTI drug — apply tighter monitoring)" if nti else ""
    if severity == "strong":
        return (
            f"Avoid combination or use with extreme caution{nti_note}. "
            "Dose adjustment likely required. Consult prescribing information."
        )
    elif severity == "moderate":
        return (
            f"Use with caution{nti_note}. "
            "Dose adjustment may be required. Monitor for adverse effects."
        )
    elif severity == "weak":
        return (
            f"Interaction unlikely to be clinically significant{nti_note}. "
            "Monitor as clinically appropriate."
        )
    else:
        return f"Minimal or no clinically relevant interaction expected{nti_note}."


# ---------------------------------------------------------------------------
# Polypharmacy DDI risk
# ---------------------------------------------------------------------------


def polypharmacy_ddi_risk(drug_list: list[dict]) -> dict:
    """
    Assess DDI risk in a polypharmacy scenario.

    Parameters
    ----------
    drug_list:
        List of drug dicts with keys:
            name (str), is_cyp_inhibitor (bool), is_cyp_inducer (bool),
            is_cyp_substrate (bool), is_nti (bool), cyp_enzyme (str).

    Returns
    -------
    dict with keys:
        total_drugs, n_interactions, high_risk_pairs, risk_level, recommendations
    """
    if not drug_list:
        return {
            "total_drugs": 0,
            "n_interactions": 0,
            "high_risk_pairs": [],
            "risk_level": "low",
            "recommendations": ["No drugs provided."],
        }

    # Validate required keys
    required_keys = {
        "name",
        "is_cyp_inhibitor",
        "is_cyp_inducer",
        "is_cyp_substrate",
        "is_nti",
        "cyp_enzyme",
    }
    for drug in drug_list:
        missing = required_keys - set(drug.keys())
        if missing:
            raise ValueError(f"Drug entry missing keys: {missing}")

    high_risk_pairs: list[tuple[str, str, str]] = []  # (perpetrator, victim, mechanism)
    n_interactions = 0
    severity_counts: dict[str, int] = {"strong": 0, "moderate": 0, "weak": 0, "minimal": 0}

    n = len(drug_list)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            perp = drug_list[i]
            victim = drug_list[j]

            # Same CYP enzyme required
            if perp["cyp_enzyme"] != victim["cyp_enzyme"]:
                continue
            if not victim["is_cyp_substrate"]:
                continue

            mechanism = None
            mock_aucr = None
            if perp["is_cyp_inhibitor"]:
                mechanism = "inhibition"
                mock_aucr = 3.0  # mock moderate inhibition
            elif perp["is_cyp_inducer"]:
                mechanism = "induction"
                mock_aucr = 0.4  # mock moderate induction

            if mechanism is None:
                continue

            n_interactions += 1
            result = classify_ddi_severity(
                aucr=mock_aucr,
                mechanism=mechanism,
                narrow_therapeutic_index=victim["is_nti"],
            )
            sev = result["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

            if sev in ("strong", "moderate"):
                high_risk_pairs.append((perp["name"], victim["name"], mechanism))

    # Deduplicate symmetric pairs for reporting
    seen: set[frozenset] = set()
    unique_pairs: list[tuple[str, str, str]] = []
    for p, v, m in high_risk_pairs:
        key = frozenset({p, v})
        if key not in seen:
            seen.add(key)
            unique_pairs.append((p, v, m))

    # Determine overall risk level
    if severity_counts.get("strong", 0) > 0:
        risk_level = "high"
    elif severity_counts.get("moderate", 0) > 0:
        risk_level = "moderate"
    else:
        risk_level = "low"

    # Build recommendations
    recommendations: list[str] = []
    if not unique_pairs:
        recommendations.append("No clinically significant DDI interactions identified.")
    else:
        for p, v, m in unique_pairs:
            recommendations.append(
                f"Review {m} interaction between {p} (perpetrator) and {v} (victim)."
            )
    if risk_level == "high":
        recommendations.append(
            "High overall risk: consider alternative therapy or intensive monitoring."
        )
    elif risk_level == "moderate":
        recommendations.append(
            "Moderate overall risk: dose adjustment or enhanced monitoring recommended."
        )

    return {
        "total_drugs": n,
        "n_interactions": n_interactions,
        "high_risk_pairs": unique_pairs,
        "risk_level": risk_level,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# DDI interaction table generator
# ---------------------------------------------------------------------------


def generate_ddi_interaction_table(
    perpetrators: list[dict],
    victims: list[dict],
) -> list[dict]:
    """
    Generate a cross-product DDI interaction table.

    Parameters
    ----------
    perpetrators:
        List of dicts with keys: name (str), mechanism (str), predicted_aucr (float).
    victims:
        List of dicts with keys: name (str), is_nti (bool).

    Returns
    -------
    List of dicts: perpetrator, victim, aucr, severity, label_warning_needed.
    """
    if not perpetrators:
        raise ValueError("perpetrators list must not be empty")
    if not victims:
        raise ValueError("victims list must not be empty")

    rows: list[dict] = []
    for perp in perpetrators:
        for victim in victims:
            aucr = perp["predicted_aucr"]
            mechanism = perp["mechanism"]
            is_nti = victim.get("is_nti", False)

            result = classify_ddi_severity(
                aucr=aucr,
                mechanism=mechanism,
                narrow_therapeutic_index=is_nti,
            )

            rows.append(
                {
                    "perpetrator": perp["name"],
                    "victim": victim["name"],
                    "aucr": aucr,
                    "severity": result["severity"],
                    "label_warning_needed": result["label_warning_needed"],
                }
            )

    return rows
