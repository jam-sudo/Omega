"""Membrane transporter module for PBPK modeling.

Implements mechanistic transporter kinetics for the most clinically relevant
drug transporters per FDA 2020 DDI guidance:

Hepatic uptake (sinusoidal):
  OATP1B1 (SLCO1B1) — statin, rifampin, repaglinide
  OATP1B3 (SLCO1B3) — similar substrates to OATP1B1

Intestinal / BBB efflux:
  P-gp (ABCB1 / MDR1) — digoxin, loperamide
  BCRP (ABCG2)        — rosuvastatin, sulfasalazine

Renal secretion:
  OCT2  (SLC22A2)  — metformin, cisplatin
  OAT1  (SLC22A6)  — NSAIDs, antivirals
  OAT3  (SLC22A8)  — penicillins, statins (minor)
  MATE1 (SLC47A1)  — metformin (tubular efflux)
  MATE2 (SLC47A2)  — metformin (tubular efflux)

All clearances (CLint) are in L/h scaled to 70 kg body weight.
Michaelis-Menten kinetics: CL_transporter = Vmax / (Km + Cu)
  where Cu = unbound concentration in the upstream compartment (mg/L).
Linear (first-order) mode: CL_transporter = CLint (constant, equivalent to
  Vmax/Km when Cu << Km, i.e. in the linear saturation regime).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "TransporterKinetics",
    "TransporterSet",
    "TransporterInhibition",
    "build_transporter_set",
]


# ---------------------------------------------------------------------------
# Per-transporter kinetics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransporterKinetics:
    """Michaelis-Menten or linear kinetic parameters for one transporter.

    Usage:
        Either provide ``clint_L_per_h`` for first-order (linear) transport, or
        ``vmax_mg_per_h`` + ``km_mg_per_L`` for saturable (Michaelis-Menten).

        The effective clearance at unbound concentration ``cu`` is::

            CL = vmax / (km + cu)          if vmax > 0
            CL = clint_L_per_h             otherwise

    Args:
        name: Human-readable transporter name.
        organ: Primary organ of action ('liver', 'gut_wall', 'kidney', 'brain').
        direction: 'uptake' (drug enters cell) or 'efflux' (drug leaves cell).
        clint_L_per_h: First-order intrinsic clearance (L/h, 70 kg).
        vmax_mg_per_h: Maximum transport rate (mg/h, 70 kg).
        km_mg_per_L: Michaelis constant for unbound drug (mg/L).
    """

    name: str
    organ: str = "liver"
    direction: str = "uptake"   # 'uptake' or 'efflux'
    clint_L_per_h: float = 0.0
    vmax_mg_per_h: float = 0.0
    km_mg_per_L: float = 1.0    # only used when vmax_mg_per_h > 0

    def effective_cl(self, cu_mg_L: float) -> float:
        """Compute effective clearance (L/h) at unbound concentration *cu*.

        Args:
            cu_mg_L: Unbound drug concentration in the upstream compartment
                     (mg/L).  Must be ≥ 0.

        Returns:
            Effective clearance (L/h) at this concentration.  For linear mode
            this is constant (= ``clint_L_per_h``).  For MM mode it decreases
            as concentration rises toward saturation.
        """
        cu = max(cu_mg_L, 0.0)
        if self.vmax_mg_per_h > 0.0:
            return self.vmax_mg_per_h / max(self.km_mg_per_L + cu, 1e-12)
        return self.clint_L_per_h

    def transport_rate(self, cu_mg_L: float) -> float:
        """Compute transport rate (mg/h) at unbound concentration *cu*.

        Args:
            cu_mg_L: Unbound drug concentration (mg/L).

        Returns:
            Transport rate (mg/h).
        """
        return self.effective_cl(cu_mg_L) * max(cu_mg_L, 0.0)


# ---------------------------------------------------------------------------
# Transporter inhibition (for DDI modifiers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransporterInhibition:
    """Inhibition of a named transporter by a perpetrator drug.

    The inhibition reduces transporter CL by the fraction::

        inhibition_fraction = [I] / ([I] + IC50)

    and thus effective CL becomes::

        CL_inhibited = CL × IC50 / (IC50 + [I])

    Args:
        transporter_name: Name of the inhibited transporter.
        ic50_uM: IC50 of the inhibitor for this transporter (µM).
        inhibitor_concentration_uM: Steady-state inhibitor concentration (µM).
    """

    transporter_name: str
    ic50_uM: float
    inhibitor_concentration_uM: float = 0.0

    @property
    def inhibition_factor(self) -> float:
        """Fraction of transporter activity remaining (0–1).

        Returns 1.0 (no inhibition) when IC50 is infinite or inhibitor is 0.
        """
        if self.ic50_uM <= 0 or self.ic50_uM == float("inf"):
            return 1.0
        c = max(self.inhibitor_concentration_uM, 0.0)
        return self.ic50_uM / max(self.ic50_uM + c, 1e-12)


# ---------------------------------------------------------------------------
# Full transporter set for a drug
# ---------------------------------------------------------------------------

@dataclass
class TransporterSet:
    """Collection of all transporter kinetics for a single drug.

    Each field corresponds to one clinically important transporter.
    ``None`` means the drug is not a substrate of that transporter.

    Inhibition entries can be registered via ``add_inhibition()`` to apply
    DDI corrections (e.g. gemfibrozil inhibiting OATP1B1 and reducing
    statin hepatic uptake).
    """

    # Hepatic uptake (sinusoidal influx)
    oatp1b1: TransporterKinetics | None = None
    oatp1b3: TransporterKinetics | None = None

    # Intestinal / BBB efflux
    pgp: TransporterKinetics | None = None
    bcrp: TransporterKinetics | None = None

    # Renal secretion
    oct2: TransporterKinetics | None = None
    oat1: TransporterKinetics | None = None
    oat3: TransporterKinetics | None = None
    mate1: TransporterKinetics | None = None
    mate2: TransporterKinetics | None = None

    # Registered inhibitions (populated at simulation time)
    _inhibitions: list[TransporterInhibition] = field(
        default_factory=list, repr=False, compare=False
    )

    def add_inhibition(self, inhibition: TransporterInhibition) -> None:
        """Register a transporter inhibition (e.g. from a co-administered DDI perpetrator)."""
        self._inhibitions.append(inhibition)

    def _inhibition_factor(self, transporter_name: str) -> float:
        """Return combined activity fraction remaining after all inhibitions."""
        factor = 1.0
        for inh in self._inhibitions:
            if inh.transporter_name == transporter_name:
                factor *= inh.inhibition_factor
        return factor

    # ------------------------------------------------------------------
    # Aggregate clearances used by body.py ODE engine
    # ------------------------------------------------------------------

    def hepatic_uptake_cl(self, cu_mg_L: float, scale: float = 1.0) -> float:
        """Combined OATP1B1 + OATP1B3 active uptake CL (L/h) at concentration *cu*.

        Args:
            cu_mg_L: Unbound drug concentration in portal blood (mg/L).
            scale: Body-weight scaling factor (BW/70).

        Returns:
            Total active hepatic uptake CL (L/h) scaled to body weight.
        """
        cl = 0.0
        for t, tname in [(self.oatp1b1, "OATP1B1"), (self.oatp1b3, "OATP1B3")]:
            if t is not None:
                f_inh = self._inhibition_factor(tname)
                cl += scale * t.effective_cl(cu_mg_L) * f_inh
        return cl

    def gut_efflux_cl(self, cu_mg_L: float, scale: float = 1.0) -> float:
        """Combined P-gp + BCRP efflux CL from gut wall (L/h) at concentration *cu*.

        Args:
            cu_mg_L: Unbound drug concentration in gut wall (mg/L).
            scale: Body-weight scaling factor.

        Returns:
            Total gut-wall efflux CL (L/h).
        """
        cl = 0.0
        for t, tname in [(self.pgp, "P-gp"), (self.bcrp, "BCRP")]:
            if t is not None:
                f_inh = self._inhibition_factor(tname)
                cl += scale * t.effective_cl(cu_mg_L) * f_inh
        return cl

    def renal_secretion_cl(self, cu_mg_L: float, scale: float = 1.0) -> float:
        """Combined active renal secretion CL (L/h) at unbound concentration *cu*.

        Includes OCT2, OAT1, OAT3, MATE1, MATE2.

        Args:
            cu_mg_L: Unbound drug concentration in kidney (mg/L).
            scale: Body-weight scaling factor.

        Returns:
            Total active renal secretion CL (L/h).
        """
        cl = 0.0
        pairs = [
            (self.oct2,  "OCT2"),
            (self.oat1,  "OAT1"),
            (self.oat3,  "OAT3"),
            (self.mate1, "MATE1"),
            (self.mate2, "MATE2"),
        ]
        for t, tname in pairs:
            if t is not None:
                f_inh = self._inhibition_factor(tname)
                cl += scale * t.effective_cl(cu_mg_L) * f_inh
        return cl

    @property
    def has_hepatic_uptake(self) -> bool:
        return self.oatp1b1 is not None or self.oatp1b3 is not None

    @property
    def has_gut_efflux(self) -> bool:
        return self.pgp is not None or self.bcrp is not None

    @property
    def has_renal_secretion(self) -> bool:
        return any(
            t is not None
            for t in (self.oct2, self.oat1, self.oat3, self.mate1, self.mate2)
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_transporter_set(
    *,
    oatp1b1_cl: float = 0.0,
    oatp1b3_cl: float = 0.0,
    pgp_cl: float = 0.0,
    bcrp_cl: float = 0.0,
    oct2_cl: float = 0.0,
    oat1_cl: float = 0.0,
    oat3_cl: float = 0.0,
    mate1_cl: float = 0.0,
    mate2_cl: float = 0.0,
) -> TransporterSet:
    """Build a TransporterSet from first-order intrinsic clearance values (L/h, 70 kg).

    Zero values are treated as "not a substrate" (transporter = None).

    Args:
        oatp1b1_cl: OATP1B1 hepatic uptake CLint (L/h).
        oatp1b3_cl: OATP1B3 hepatic uptake CLint (L/h).
        pgp_cl:     P-gp gut-wall efflux CLint (L/h).
        bcrp_cl:    BCRP gut-wall efflux CLint (L/h).
        oct2_cl:    OCT2 renal secretion CLint (L/h).
        oat1_cl:    OAT1 renal secretion CLint (L/h).
        oat3_cl:    OAT3 renal secretion CLint (L/h).
        mate1_cl:   MATE1 renal secretion CLint (L/h).
        mate2_cl:   MATE2 renal secretion CLint (L/h).

    Returns:
        TransporterSet with the specified transporters active.
    """
    def _make(name, organ, direction, cl):
        return TransporterKinetics(
            name=name, organ=organ, direction=direction, clint_L_per_h=cl
        )

    return TransporterSet(
        oatp1b1=_make("OATP1B1", "liver", "uptake", oatp1b1_cl) if oatp1b1_cl > 0 else None,
        oatp1b3=_make("OATP1B3", "liver", "uptake", oatp1b3_cl) if oatp1b3_cl > 0 else None,
        pgp=_make("P-gp", "gut_wall", "efflux", pgp_cl) if pgp_cl > 0 else None,
        bcrp=_make("BCRP", "gut_wall", "efflux", bcrp_cl) if bcrp_cl > 0 else None,
        oct2=_make("OCT2", "kidney", "uptake", oct2_cl) if oct2_cl > 0 else None,
        oat1=_make("OAT1", "kidney", "uptake", oat1_cl) if oat1_cl > 0 else None,
        oat3=_make("OAT3", "kidney", "uptake", oat3_cl) if oat3_cl > 0 else None,
        mate1=_make("MATE1", "kidney", "efflux", mate1_cl) if mate1_cl > 0 else None,
        mate2=_make("MATE2", "kidney", "efflux", mate2_cl) if mate2_cl > 0 else None,
    )
