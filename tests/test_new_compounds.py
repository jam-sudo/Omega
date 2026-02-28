"""Tests for ibuprofen and atenolol reference compounds.

Validates that PBPK simulations produce physiologically plausible PK
for newly added compounds with wide tolerance ranges.
"""

import numpy as np

from omega_pbpk._compat import np_trapz
from omega_pbpk.config import load_compound
from omega_pbpk.core.body import WholeBodyPBPK
from omega_pbpk.drugs.atenolol import ATENOLOL
from omega_pbpk.drugs.ibuprofen import IBUPROFEN


def _simulate_oral(drug, dose_mg, t_end_h, dt_h=0.05):
    """Run oral simulation and return PK summary."""
    model = WholeBodyPBPK(drug, body_weight=70.0)
    model.setup_oral(dose_mg)
    result = model.simulate(t_end_h=t_end_h, dt_h=dt_h)
    conc = result.plasma_concentration()
    time = result.time_h
    cmax = float(np.max(conc))
    auc = float(np_trapz(conc, time))
    return {"Cmax": cmax, "AUC": auc, "conc": conc, "time": time}


class TestIbuprofen:
    """Ibuprofen 200 mg oral, 12 h simulation."""

    def test_simulation_runs(self):
        """Ibuprofen simulation completes without error."""
        pk = _simulate_oral(IBUPROFEN, dose_mg=200.0, t_end_h=12.0)
        assert pk["Cmax"] > 0
        assert pk["AUC"] > 0

    def test_cmax_range(self):
        """Ibuprofen Cmax in [5, 40] mg/L (literature ~15-25 mg/L for 200 mg)."""
        pk = _simulate_oral(IBUPROFEN, dose_mg=200.0, t_end_h=12.0)
        assert 5.0 <= pk["Cmax"] <= 40.0, f"Ibuprofen Cmax={pk['Cmax']:.2f} mg/L outside [5, 40]"

    def test_auc_range(self):
        """Ibuprofen AUC in [20, 200] mg*h/L."""
        pk = _simulate_oral(IBUPROFEN, dose_mg=200.0, t_end_h=12.0)
        assert 20.0 <= pk["AUC"] <= 200.0, f"Ibuprofen AUC={pk['AUC']:.2f} mg*h/L outside [20, 200]"

    def test_yaml_loads(self):
        """Compound YAML loads correctly via load_compound."""
        from pathlib import Path

        drug = load_compound(Path("compounds/ibuprofen.yaml"))
        assert drug.name == "Ibuprofen"
        assert abs(drug.mw - 206.28) < 0.01
        assert drug.drug_type == "monoprotic_acid"


class TestAtenolol:
    """Atenolol 100 mg oral, 24 h simulation."""

    def test_simulation_runs(self):
        """Atenolol simulation completes without error."""
        pk = _simulate_oral(ATENOLOL, dose_mg=100.0, t_end_h=24.0)
        assert pk["Cmax"] > 0
        assert pk["AUC"] > 0

    def test_cmax_range(self):
        """Atenolol Cmax in [0.05, 5.0] mg/L (literature ~0.3-0.6 mg/L for 100 mg).

        Wide upper bound: PBPK models may overpredict for low-perm BCS III drugs
        due to incomplete absorption modelling fidelity.
        """
        pk = _simulate_oral(ATENOLOL, dose_mg=100.0, t_end_h=24.0)
        assert 0.05 <= pk["Cmax"] <= 5.0, f"Atenolol Cmax={pk['Cmax']:.2f} mg/L outside [0.05, 5.0]"

    def test_auc_range(self):
        """Atenolol AUC in [0.5, 10] mg*h/L."""
        pk = _simulate_oral(ATENOLOL, dose_mg=100.0, t_end_h=24.0)
        assert 0.5 <= pk["AUC"] <= 10.0, f"Atenolol AUC={pk['AUC']:.2f} mg*h/L outside [0.5, 10]"

    def test_yaml_loads(self):
        """Compound YAML loads correctly via load_compound."""
        from pathlib import Path

        drug = load_compound(Path("compounds/atenolol.yaml"))
        assert drug.name == "Atenolol"
        assert abs(drug.mw - 266.34) < 0.01
        assert drug.drug_type == "monoprotic_base"
