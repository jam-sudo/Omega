"""Tests for pediatric CYP ontogeny and physiological maturation functions."""


import pytest

from omega_pbpk.clinical.ontogeny import (
    cyp1a2_ontogeny,
    cyp2d6_ontogeny,
    cyp3a4_ontogeny,
    get_pediatric_scaling,
    gfr_ontogeny,
)


class TestCYP3A4Ontogeny:
    def test_birth_near_basal(self):
        """At birth (age=0), CYP3A4 should be near basal fraction (0.1)."""
        f = cyp3a4_ontogeny(0.0)
        assert f == pytest.approx(0.1, abs=0.05)

    def test_adult_approaches_1(self):
        """At 18+ years, activity should be ~1.0 (adult)."""
        f = cyp3a4_ontogeny(18.0)
        assert f == pytest.approx(1.0, abs=0.01)

    def test_monotone_increasing(self):
        """Activity should increase monotonically from birth to adult."""
        ages = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 18.0]
        activities = [cyp3a4_ontogeny(a) for a in ages]
        for i in range(1, len(activities)):
            assert activities[i] >= activities[i - 1]

    def test_at_tm50_approximately_midpoint(self):
        """At tm50=0.25y: activity ≈ basal + 0.5*(1-basal) = 0.55."""
        f = cyp3a4_ontogeny(0.25)
        assert f == pytest.approx(0.55, abs=0.05)

    def test_in_01_range(self):
        for age in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 18.0]:
            f = cyp3a4_ontogeny(age)
            assert 0.0 <= f <= 1.0

    def test_negative_age_treated_as_zero(self):
        """Negative age should clamp to 0."""
        assert cyp3a4_ontogeny(-1.0) == cyp3a4_ontogeny(0.0)

    def test_infant_less_than_adult(self):
        assert cyp3a4_ontogeny(0.1) < cyp3a4_ontogeny(18.0)


class TestCYP2D6Ontogeny:
    def test_birth_near_basal(self):
        """At birth, CYP2D6 ≈ 2% activity."""
        f = cyp2d6_ontogeny(0.0)
        assert f == pytest.approx(0.02, abs=0.02)

    def test_adult_approaches_1(self):
        f = cyp2d6_ontogeny(18.0)
        assert f == pytest.approx(1.0, abs=0.02)

    def test_monotone_increasing(self):
        ages = [0.0, 0.1, 0.4, 1.0, 3.0, 10.0, 18.0]
        activities = [cyp2d6_ontogeny(a) for a in ages]
        for i in range(1, len(activities)):
            assert activities[i] >= activities[i - 1]

    def test_lower_than_cyp3a4_at_birth(self):
        """CYP2D6 has lower basal than CYP3A4 at birth."""
        assert cyp2d6_ontogeny(0.0) < cyp3a4_ontogeny(0.0)

    def test_in_01_range(self):
        for age in [0.0, 0.1, 0.4, 1.0, 5.0]:
            assert 0.0 <= cyp2d6_ontogeny(age) <= 1.0


class TestCYP1A2Ontogeny:
    def test_birth_very_low(self):
        """CYP1A2 is essentially absent at birth (≈1%)."""
        f = cyp1a2_ontogeny(0.0)
        assert f == pytest.approx(0.01, abs=0.01)

    def test_adult_approaches_1(self):
        f = cyp1a2_ontogeny(18.0)
        assert f == pytest.approx(1.0, abs=0.02)

    def test_slower_maturation_than_cyp3a4(self):
        """CYP1A2 (tm50=1.0y) matures slower than CYP3A4 (tm50=0.25y) at 0.5y."""
        f1a2 = cyp1a2_ontogeny(0.5)
        f3a4 = cyp3a4_ontogeny(0.5)
        assert f1a2 < f3a4

    def test_monotone_increasing(self):
        ages = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 18.0]
        acts = [cyp1a2_ontogeny(a) for a in ages]
        for i in range(1, len(acts)):
            assert acts[i] >= acts[i - 1]

    def test_in_01_range(self):
        for age in [0.0, 0.5, 1.0, 5.0, 18.0]:
            assert 0.0 <= cyp1a2_ontogeny(age) <= 1.0


class TestGFROntogeny:
    def test_birth_near_basal(self):
        """Neonatal GFR ≈ 10% of adult."""
        f = gfr_ontogeny(0.0)
        assert f == pytest.approx(0.1, abs=0.05)

    def test_adult_approaches_1(self):
        f = gfr_ontogeny(5.0)
        assert f == pytest.approx(1.0, abs=0.02)

    def test_monotone_increasing(self):
        ages = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        acts = [gfr_ontogeny(a) for a in ages]
        for i in range(1, len(acts)):
            assert acts[i] >= acts[i - 1]

    def test_in_01_range(self):
        for age in [0.0, 0.1, 0.5, 1.0, 5.0]:
            assert 0.0 <= gfr_ontogeny(age) <= 1.0

    def test_one_year_above_50pct(self):
        """By age 1, GFR should exceed 50% of adult."""
        assert gfr_ontogeny(1.0) > 0.5


class TestGetPediatricScaling:
    def setup_method(self):
        self.neonate = get_pediatric_scaling(age_years=0.0, weight_kg=3.5)
        self.toddler = get_pediatric_scaling(age_years=2.0, weight_kg=12.0)
        self.adult = get_pediatric_scaling(age_years=25.0, weight_kg=70.0)

    def test_returns_dict(self):
        assert isinstance(self.neonate, dict)

    def test_required_keys(self):
        keys = {"cyp3a4", "cyp2d6", "cyp1a2", "gfr", "hepatic_blood_flow", "cardiac_output"}
        assert keys.issubset(self.neonate.keys())

    def test_all_positive(self):
        for v in self.neonate.values():
            assert v > 0

    def test_adult_cyp_near_1(self):
        assert self.adult["cyp3a4"] == pytest.approx(1.0, abs=0.02)
        assert self.adult["cyp2d6"] == pytest.approx(1.0, abs=0.02)
        assert self.adult["cyp1a2"] == pytest.approx(1.0, abs=0.02)

    def test_adult_flow_near_1(self):
        """Adult (70kg) allometric flow scale should be 1.0."""
        assert self.adult["hepatic_blood_flow"] == pytest.approx(1.0, rel=1e-4)
        assert self.adult["cardiac_output"] == pytest.approx(1.0, rel=1e-4)

    def test_neonate_cyp_less_than_adult(self):
        for cyp in ["cyp3a4", "cyp2d6", "cyp1a2"]:
            assert self.neonate[cyp] < self.adult[cyp]

    def test_neonate_gfr_less_than_adult(self):
        assert self.neonate["gfr"] < self.adult["gfr"]

    def test_neonate_flow_less_than_adult(self):
        """Neonate (3.5 kg) has lower allometric flow than adult (70 kg)."""
        assert self.neonate["hepatic_blood_flow"] < self.adult["hepatic_blood_flow"]

    def test_toddler_between_neonate_and_adult(self):
        for key in ["cyp3a4", "gfr", "hepatic_blood_flow"]:
            assert self.neonate[key] <= self.toddler[key] <= self.adult[key] + 0.01

    def test_allometric_scale_formula(self):
        """hepatic_blood_flow = (weight/70)^0.75."""
        f = get_pediatric_scaling(0.0, 12.0)
        expected = (12.0 / 70.0) ** 0.75
        assert f["hepatic_blood_flow"] == pytest.approx(expected, rel=1e-6)
