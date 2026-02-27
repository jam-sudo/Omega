"""Clinical tools — dose optimization, FIH, multi-dose, formulation comparison."""

from omega_pbpk.clinical.allometry import (
    AllometricPrediction,
    predict_human_from_preclinical,
    scale_multi_species,
    scale_single_species,
)
from omega_pbpk.clinical.ddi_report import (
    KDEG,
    DDIInhibitor,
    DDIRiskReport,
    assess_ddi_risk,
    format_report,
)
from omega_pbpk.clinical.disease import (
    ChildPughClass,
    DiseaseState,
    HeartFailure,
    HepaticImpairment,
    NyhaClass,
    ObesityState,
    PregnancyState,
    RenalImpairment,
    RenalImpairmentStage,
    ScaledParameters,
    TrimesterStage,
    apply_disease_scaling,
)
from omega_pbpk.clinical.dose_optimization import (
    DoseOptimizer,
    FormulationComparator,
    MultiDoseSimulator,
)
from omega_pbpk.clinical.ivive import (
    IVIVEResult,
    estimate_clint_for_target_clh,
    scale_hepatocyte_clint,
    scale_microsomal_clint,
)
from omega_pbpk.clinical.nca import (
    NCAResult,
    run_nca,
)
from omega_pbpk.clinical.ontogeny import (
    cyp1a2_ontogeny,
    cyp2d6_ontogeny,
    cyp3a4_ontogeny,
    get_pediatric_scaling,
    gfr_ontogeny,
)
from omega_pbpk.clinical.pgx_pbpk import (
    PGxPBPKResult,
    pgx_report_html,
    plot_pgx_forest,
    run_pgx_pbpk,
)
from omega_pbpk.clinical.report import ReportInput, generate_report, quick_report
from omega_pbpk.clinical.trial import (
    ArmPKSummary,
    BioequivalenceResult,
    DoseRegimen,
    SubjectPKResult,
    TrialArm,
    TrialResult,
    compute_bioequivalence,
    run_clinical_trial,
    simulate_arm,
)

__all__ = [
    "RenalImpairment",
    "RenalImpairmentStage",
    "HepaticImpairment",
    "ChildPughClass",
    "HeartFailure",
    "NyhaClass",
    "ObesityState",
    "PregnancyState",
    "TrimesterStage",
    "DiseaseState",
    "ScaledParameters",
    "apply_disease_scaling",
    "DoseOptimizer",
    "MultiDoseSimulator",
    "FormulationComparator",
    "cyp3a4_ontogeny",
    "cyp2d6_ontogeny",
    "cyp1a2_ontogeny",
    "gfr_ontogeny",
    "get_pediatric_scaling",
    "DDIInhibitor",
    "DDIRiskReport",
    "KDEG",
    "assess_ddi_risk",
    "format_report",
    "NCAResult",
    "run_nca",
    "ReportInput",
    "generate_report",
    "quick_report",
    "AllometricPrediction",
    "scale_single_species",
    "scale_multi_species",
    "predict_human_from_preclinical",
    "IVIVEResult",
    "scale_microsomal_clint",
    "scale_hepatocyte_clint",
    "estimate_clint_for_target_clh",
    "PGxPBPKResult",
    "run_pgx_pbpk",
    "plot_pgx_forest",
    "pgx_report_html",
    # Phase 15: Clinical trial simulation
    "DoseRegimen",
    "TrialArm",
    "SubjectPKResult",
    "ArmPKSummary",
    "BioequivalenceResult",
    "TrialResult",
    "simulate_arm",
    "compute_bioequivalence",
    "run_clinical_trial",
]
