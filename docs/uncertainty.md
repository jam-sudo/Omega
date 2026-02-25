# Uncertainty Propagation — Omega PBPK v0.7

## Monte Carlo virtual population

`omega_pbpk.population.physiology.VirtualPopulation` generates N virtual subjects by
sampling physiological parameters from log-normal distributions around ICRP reference values.

### Usage

```python
from omega_pbpk.population.physiology import VirtualPopulation

pop = VirtualPopulation(n=100, seed=42)
subjects = pop.generate()
stats = pop.summary_stats(subjects)
```

### Sampled parameters per subject

| Parameter | Reference | CV |
|-----------|-----------|-----|
| Body weight | 70 kg | 0.15 |
| Cardiac output | 390 L/h (× BW^0.75 scaling) | 0.10 |
| GFR | 7.5 L/h (× BW^0.75 scaling) | 0.15 |
| Liver weight | 1800 g (× BW scaling) | 0.15 |
| MPPGL | 40 pmol/mg | 0.30 |
| Organ volumes | ICRP reference × BW scaling | organ-specific |
| Organ flows | ICRP reference (renormalized to CO) | organ-specific |

### Pharmacogenomic variability

Additional inter-individual variability via CYP polymorphism:

```python
from omega_pbpk.pharmacogenomics.cyp_polymorphism import PGxAnalyzer

analyzer = PGxAnalyzer()
results = analyzer.analyze_gene("CYP2D6", "Caucasian")
# Each result has clint_scaling_factor and population_frequency
```

### Population PK simulation

Run the PBPK model for each virtual subject to generate a population PK distribution,
then visualize with median and percentile bands:

```python
from omega_pbpk.visualization.plots import PKPlotter

plotter = PKPlotter()
plotter.plot_population(time_h, cp_matrix, save_path="pop_pk.png")
```
