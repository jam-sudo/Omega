"""Tests for drug_interaction_score module."""

import pytest

from omega_pbpk.risk.drug_interaction_score import (
    DDIRiskEntry,
    DDIRiskReport,
    score_ddi_risk,
    screen_combinations,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def make_report(
    primary_drug="PrimaryDrug",
    cyp3a4_substrate=False,
    cyp2d6_substrate=False,
    pgp_substrate=False,
    interacting_drugs=None,
):
    return score_ddi_risk(
        primary_drug=primary_drug,
        primary_cyp3a4_substrate=cyp3a4_substrate,
        primary_cyp2d6_substrate=cyp2d6_substrate,
        primary_pgp_substrate=pgp_substrate,
        interacting_drugs=interacting_drugs or [],
    )


# ── basic construction ────────────────────────────────────────────────────────


def test_returns_ddi_risk_report():
    report = make_report()
    assert isinstance(report, DDIRiskReport)


def test_primary_drug_name_preserved():
    report = make_report(primary_drug="Alprazolam")
    assert report.primary_drug == "Alprazolam"


def test_empty_interacting_drugs_low_risk():
    report = make_report()
    assert report.overall_risk == "low"
    assert report.max_risk_score == 0.0
    assert report.n_major_interactions == 0


def test_empty_interactions_list():
    report = make_report()
    assert report.interactions == []


# ── CYP3A4 inhibition ────────────────────────────────────────────────────────


def test_strong_cyp3a4_inhibitor_with_substrate_gives_major():
    # Ki = 0.1 uM → AUCR = 1 + 1/0.1 = 11, score = min(50, 10*10) = 50
    report = score_ddi_risk(
        primary_drug="Midazolam",
        primary_cyp3a4_substrate=True,
        interacting_drugs=[{"name": "Ketoconazole", "cyp3a4_inhibitor_ki": 0.1}],
    )
    assert len(report.interactions) == 1
    entry = report.interactions[0]
    assert isinstance(entry, DDIRiskEntry)
    assert entry.severity in ("major", "contraindicated")
    assert entry.risk_score >= 50.0


def test_weak_cyp3a4_inhibitor_gives_low_risk():
    # Ki = 100 uM → AUCR = 1.01, score ≈ 0.1 (minor)
    report = score_ddi_risk(
        primary_drug="Midazolam",
        primary_cyp3a4_substrate=True,
        interacting_drugs=[{"name": "WeakInhibitor", "cyp3a4_inhibitor_ki": 100.0}],
    )
    entry = report.interactions[0]
    assert entry.severity in ("none", "minor")
    assert entry.risk_score < 10.0


def test_cyp3a4_inhibitor_without_substrate_no_interaction():
    # Primary is NOT a substrate → inhibition irrelevant
    report = score_ddi_risk(
        primary_drug="PrimaryDrug",
        primary_cyp3a4_substrate=False,
        interacting_drugs=[{"name": "Ketoconazole", "cyp3a4_inhibitor_ki": 0.1}],
    )
    entry = report.interactions[0]
    assert entry.severity == "none"
    assert entry.risk_score == 0.0


def test_mechanism_is_cyp_inhibition_for_cyp3a4():
    report = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=[{"name": "Inh", "cyp3a4_inhibitor_ki": 1.0}],
    )
    assert report.interactions[0].mechanism == "cyp_inhibition"


# ── CYP2D6 inhibition ────────────────────────────────────────────────────────


def test_cyp2d6_inhibition_increases_risk_score():
    report = score_ddi_risk(
        primary_drug="Codeine",
        primary_cyp2d6_substrate=True,
        interacting_drugs=[{"name": "Fluoxetine", "cyp2d6_inhibitor_ki": 0.5}],
    )
    entry = report.interactions[0]
    assert entry.risk_score > 0.0
    assert entry.mechanism == "cyp_inhibition"


# ── P-gp transporter ─────────────────────────────────────────────────────────


def test_pgp_inhibitor_increases_risk_for_pgp_substrate():
    report = score_ddi_risk(
        primary_drug="Digoxin",
        primary_pgp_substrate=True,
        interacting_drugs=[{"name": "Verapamil", "pgp_inhibitor": True}],
    )
    entry = report.interactions[0]
    assert entry.risk_score >= 15.0
    assert entry.mechanism == "transporter"


def test_pgp_inhibitor_without_pgp_substrate_no_risk():
    report = score_ddi_risk(
        primary_drug="DrugA",
        primary_pgp_substrate=False,
        interacting_drugs=[{"name": "Verapamil", "pgp_inhibitor": True}],
    )
    entry = report.interactions[0]
    assert entry.risk_score == 0.0


# ── severity thresholds ───────────────────────────────────────────────────────


def test_severity_none_for_zero_score():
    report = make_report(interacting_drugs=[{"name": "Inert"}])
    assert report.interactions[0].severity == "none"


def test_severity_minor_10_to_30():
    # Ki = 1 uM for CYP3A4: AUCR = 1 + 1/1 = 2 → score = (2-1)*10 = 10 → minor
    report2 = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=[{"name": "MildInh", "cyp3a4_inhibitor_ki": 1.0}],
    )
    entry2 = report2.interactions[0]
    assert entry2.severity == "minor"
    assert entry2.risk_score == pytest.approx(10.0)


# ── overall risk ──────────────────────────────────────────────────────────────


def test_multiple_major_interactions_critical():
    # Two major interactions → critical
    drugs = [
        {"name": "InhA", "cyp3a4_inhibitor_ki": 0.1},
        {"name": "InhB", "cyp3a4_inhibitor_ki": 0.1},
    ]
    report = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=drugs,
    )
    assert report.n_major_interactions >= 2
    assert report.overall_risk == "critical"


def test_single_major_high_risk():
    # One major interaction → "high"
    report = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=[{"name": "InhA", "cyp3a4_inhibitor_ki": 0.1}],
    )
    # score=50 → major → high (not critical since n_major=1)
    assert report.overall_risk == "high"


def test_n_major_interactions_counts_correctly():
    # InhA: Ki=0.1 → AUCR=11 → score=50 → major
    # Inert: score=0 → none
    # InhB: Ki=0.5 → AUCR=3 → score=20 → moderate (not major)
    drugs = [
        {"name": "InhA", "cyp3a4_inhibitor_ki": 0.1},  # score 50 → major
        {"name": "Inert"},  # score 0 → none
        {"name": "InhB", "cyp3a4_inhibitor_ki": 0.5},  # score 20 → moderate
    ]
    report = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=drugs,
    )
    # Only first is major/contraindicated
    assert report.n_major_interactions == 1


# ── priority alert ────────────────────────────────────────────────────────────


def test_priority_alert_no_interactions():
    report = make_report()
    assert "No interactions" in report.priority_alert


def test_priority_alert_identifies_worst():
    drugs = [
        {"name": "Mild", "cyp3a4_inhibitor_ki": 10.0},
        {"name": "Strong", "cyp3a4_inhibitor_ki": 0.1},
    ]
    report = score_ddi_risk(
        primary_drug="Sub",
        primary_cyp3a4_substrate=True,
        interacting_drugs=drugs,
    )
    assert "Strong" in report.priority_alert


# ── screen_combinations ───────────────────────────────────────────────────────


def test_screen_combinations_returns_list():
    candidates = [
        {"name": "DrugA", "cyp3a4_inhibitor_ki": 0.1},
        {"name": "DrugB"},
        {"name": "DrugC", "cyp3a4_inhibitor_ki": 2.0},
    ]
    reports = screen_combinations(
        primary_drug="Sub",
        primary_drug_props={"cyp3a4_substrate": True},
        candidate_drugs=candidates,
    )
    assert isinstance(reports, list)
    assert len(reports) == 3


def test_screen_combinations_sorted_descending():
    candidates = [
        {"name": "Weak", "cyp3a4_inhibitor_ki": 20.0},
        {"name": "Strong", "cyp3a4_inhibitor_ki": 0.1},
        {"name": "Inert"},
    ]
    reports = screen_combinations(
        primary_drug="Sub",
        primary_drug_props={"cyp3a4_substrate": True},
        candidate_drugs=candidates,
    )
    scores = [r.max_risk_score for r in reports]
    assert scores == sorted(scores, reverse=True)


def test_screen_combinations_strongest_first():
    candidates = [
        {"name": "Weak", "cyp3a4_inhibitor_ki": 20.0},
        {"name": "Strong", "cyp3a4_inhibitor_ki": 0.1},
    ]
    reports = screen_combinations(
        primary_drug="Sub",
        primary_drug_props={"cyp3a4_substrate": True},
        candidate_drugs=candidates,
    )
    assert reports[0].primary_drug == "Sub"
    # Strong inhibitor has higher risk score → comes first
    assert reports[0].max_risk_score > reports[1].max_risk_score
