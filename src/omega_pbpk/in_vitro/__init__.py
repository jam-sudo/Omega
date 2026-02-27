"""In vitro assay simulators and IVIVE pipeline.

Modules
-------
assay
    Microsomal and hepatocyte metabolic stability assay simulators.
    Provides substrate-depletion ODE models and whole-liver CLint scaling.

caco2
    Caco-2 bidirectional permeability assay simulator.
    Models A→B / B→A transport, efflux ratio, and P-gp/BCRP flagging.

receptor
    Receptor occupancy and functional effect models.
    Includes equilibrium (Langmuir), kinetic (kon/koff ODE), and the
    Black-Leff operational model.  EC50Extractor fits a Hill equation
    to dose-response data.

pipeline
    Full in vitro → in vivo extrapolation (IVIVE) pipeline.
    SMILES → ADME → Phase 2 → Assay → Caco-2 → IVIVE → Drug object.

Quick start::

    from omega_pbpk.in_vitro import InVitroPipeline, InVitroRequest

    req = InVitroRequest(
        smiles="Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
        dose_mg=100.0,
        assay_type="hepatocyte",
        run_caco2=True,
    )
    result = InVitroPipeline().run(req)
    print(f"CLh = {result.clh_l_per_h:.3f} L/h")
    print(f"Drug ready for PBPK: {result.drug.name}")
"""

from omega_pbpk.in_vitro.assay import (
    HPGL,
    LIVER_WEIGHT_G,
    MPPGL,
    AssayResult,
    HepatocyteAssay,
    MicrosomesAssay,
)
from omega_pbpk.in_vitro.caco2 import (
    MINI_SURFACE_AREA_CM2,
    STANDARD_SURFACE_AREA_CM2,
    Caco2Assay,
    Caco2Result,
)
from omega_pbpk.in_vitro.pipeline import InVitroPipeline, InVitroRequest, InVitroResult
from omega_pbpk.in_vitro.receptor import (
    EC50Extractor,
    EC50Result,
    EquilibriumOccupancy,
    KineticOccupancy,
    OperationalModel,
    OccupancyResult,
)

__all__ = [
    # assay
    "AssayResult",
    "MicrosomesAssay",
    "HepatocyteAssay",
    "MPPGL",
    "HPGL",
    "LIVER_WEIGHT_G",
    # caco2
    "Caco2Result",
    "Caco2Assay",
    "STANDARD_SURFACE_AREA_CM2",
    "MINI_SURFACE_AREA_CM2",
    # receptor
    "OccupancyResult",
    "EC50Result",
    "EquilibriumOccupancy",
    "KineticOccupancy",
    "OperationalModel",
    "EC50Extractor",
    # pipeline
    "InVitroRequest",
    "InVitroResult",
    "InVitroPipeline",
]
