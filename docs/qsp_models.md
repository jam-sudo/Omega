# QSP models

## Turnover biomarker (`turnover`)

State:

- `B_biomarker`

Equation:

\[
\frac{dB}{dt} = k_{in} - k_{out} B - E(C_{signal}) B
\]

Drug effect:

\[
E(C) = \frac{E_{max} C^{h}}{EC50^{h} + C^{h}}
\]

Signal source (`signal`):

- `plasma_total` (default)
- `plasma_unbound`

Parameters (`params`):

- `kin`
- `kout`
- `emax`
- `ec50_mg_per_L`
- `hill` (optional, default 1.0)
- `B0` (optional; default `kin/kout`)

Non-negativity handling:

- state is clamped to non-negative after solve
- RHS prevents negative drift when `B=0`
