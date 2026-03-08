# ML Training Data

Synthetic PK training data generated from the Omega PBPK engine.

## Regenerating Data

```bash
# Generate both 1-cpt and PBPK data (default: 500 PBPK + 100K 1-cpt samples)
python -m omega_pbpk.ml.data.synthetic

# Generate only PBPK ODE data (50K samples for production)
python -m omega_pbpk.ml.data.synthetic --mode pbpk --n-samples 50000

# Generate only 1-compartment analytical data
python -m omega_pbpk.ml.data.synthetic --mode 1cpt --n-1cpt 100000

# Custom output directory and seed
python -m omega_pbpk.ml.data.synthetic --output-dir data/ml --seed 42 --n-workers 8
```

## Parameter Ranges

### PBPK ODE (6D parameter space)

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| logP | [-2, 6] | Linear | Octanol-water partition coefficient |
| fup | [0.01, 1.0] | Log-uniform | Fraction unbound in plasma |
| clint_L_h | [0.1, 1000] | Log-uniform | Intrinsic clearance (L/h) |
| mw | [150, 900] | Log-uniform | Molecular weight (g/mol) |
| rbp | [0.3, 3.0] | Log-uniform | Blood-to-plasma ratio |
| peff | [0.01, 100] | Log-uniform | Effective permeability (x1e-4 cm/s) |

### 1-Compartment Analytical (5D parameter space)

| Parameter | Range | Scale | Description |
|-----------|-------|-------|-------------|
| ka | [0.1, 5.0] | Log-uniform | Absorption rate constant (/h) |
| ke | [0.01, 2.0] | Log-uniform | Elimination rate constant (/h) |
| Vd | [5, 500] | Log-uniform | Volume of distribution (L) |
| F | [0.1, 1.0] | Uniform | Bioavailability |
| dose | [1, 1000] | Log-uniform | Dose (mg) |

## Output Format

HDF5 files with gzip compression:

- `params`: Input parameter vectors, shape (n_samples, n_params), float64
- `curves`: Plasma concentration-time curves, shape (n_samples, 241), float64
- `metrics`: Scalar PK metrics [Cmax, AUC, Tmax, t1/2], shape (n_samples, 4), float64
- `time_h`: Time vector, shape (241,), float64 (0 to 24h at 0.1h intervals)
- Attributes: `param_names`, `metric_names`, `metadata` (JSON-encoded)

## Expected File Sizes

| Dataset | Samples | File Size |
|---------|---------|-----------|
| 1-cpt (100K) | 100,000 | ~150 MB |
| PBPK (500) | ~400-450 | ~1 MB |
| PBPK (50K) | ~40K-45K | ~80 MB |

## Data Versioning

Data files are excluded from git via `.gitignore`. To version data:
1. Record the generation command, seed, and commit hash in experiment logs
2. Store HDF5 metadata (embedded in each file) for reproducibility
3. Use consistent seeds across runs for reproducible datasets

The generation metadata (date, params, seed) is embedded in each HDF5 file
and can be read via `CurveDataset.load_hdf5(path).metadata`.

## Multi-Fidelity Training Strategy

1. **Pre-train** on 1-cpt analytical data (100K samples, instant generation)
2. **Fine-tune** on PBPK ODE data (50K samples, ~7h generation time)
3. **Validate** against held-out ODE data (AAFE target: < 1.5)
