# Synthetic benchmark data

The complete benchmark used in the MCDD experiments is generated synthetically and is **not stored in this GitHub repository** because the resulting HDF5 files are large.

The benchmark contains:

- 3 drift types: abrupt, gradual, and incremental;
- 3 distributions: normal, exponential, and gamma;
- 1,000 replications for each drift/distribution combination;
- 70,000 observations per stream;
- drift starting at observation 40,000;
- additive Gaussian noise with standard deviation 0.1;
- gradual and incremental transition lengths selected from 1,000, 2,000, or 3,000 observations;
- seeds 42 through 1041 for the 1,000 replications of each scenario.

## Generate the complete benchmark

From the repository root, with the `MCDD` Conda environment activated, run:

```bash
python scripts/generate_datasets.py
```

This generates nine HDF5 archives under `data/datasets/`:

```text
abrupt_normal.h5
abrupt_exponential.h5
abrupt_gamma.h5
gradual_normal.h5
gradual_exponential.h5
gradual_gamma.h5
incremental_normal.h5
incremental_exponential.h5
incremental_gamma.h5
```

Each archive stores the 1,000 streams together with the seed, drift limits, transition length, and distribution parameters required to interpret each replication.

A reduced generation check can be run with:

```bash
python scripts/generate_datasets.py --replications 2 --output-dir data/test
```

The generated `.h5` files are intentionally ignored by Git. This keeps the repository lightweight while preserving reproducibility through the generation script and fixed random seeds.
