# Multiple Contrast Drift Detection (MCDD)

This repository contains the synthetic data generators, detector implementation, and experiment workflow used to evaluate **Multiple Contrast Drift Detection (MCDD)**.

The repository includes:

- generators for abrupt, gradual, and incremental concept drift;
- normal, exponential, and gamma data streams;
- reproducible HDF5 generation for the complete 9,000-stream benchmark;
- a visual dataset-generation notebook;
- the official `MCDD` detector implementation;
- Traditional Single Hypothesis, KSWIN, and LORD-LD baselines;
- a notebook for focused and complete experiment execution;
- lightweight automated software tests.

SEED is not included in the current experiment implementation.

## Repository structure

```text
.
├── .gitignore
├── CITATION.cff
├── LICENSE
├── environment.yml
├── data/
│   ├── README.md
│   └── datasets/              # generated locally; .h5 files are not versioned
├── notebooks/
│   ├── dataset_generation_demo.ipynb
│   └── run_experiments.ipynb
├── results/
│   └── README.md              # generated CSV files are not versioned yet
├── scripts/
│   └── generate_datasets.py
├── src/
│   └── mcdd/
│       ├── datasets/
│       │   └── generators.py
│       ├── detectors/
│       │   ├── mcdd.py
│       │   └── lord.py
│       └── experiments/
│           └── evaluation.py
└── tests/
    ├── conftest.py
    ├── test_generators.py
    ├── test_mcdd.py
    └── test_evaluation.py
```

## Environment

The supplied `environment.yml` reproduces the Conda environment used for the experiments:

```bash
conda env create -f environment.yml
conda activate MCDD
```

The environment was exported from a Windows x86-64 setup, so it contains some platform-specific Conda packages. The machine-specific Conda prefix was removed. `h5py` is included for the benchmark archives and `pytest` for the automated tests.

## Synthetic benchmark

The complete benchmark contains:

- 3 drift types: abrupt, gradual, and incremental;
- 3 distributions: normal, exponential, and gamma;
- 1,000 replications for each drift/distribution combination;
- 70,000 observations per stream;
- drift starting at observation 40,000;
- additive Gaussian noise with standard deviation 0.1;
- transition lengths selected from 1,000, 2,000, or 3,000 observations for gradual and incremental drift;
- seeds 42 through 1041 for the 1,000 replications of each scenario.

Gradual and incremental transitions are generated from the selected distribution. In particular, exponential and gamma transition intervals are not generated from a normal distribution.

### Why the HDF5 files are not stored in GitHub

The complete `.h5` benchmark files are intentionally **not included in the repository because of their size**. They can be reproduced from the fixed generation code and random seeds provided here.

Generate all nine archives from the repository root with:

```bash
python scripts/generate_datasets.py
```

This creates:

```text
data/datasets/
├── abrupt_normal.h5
├── abrupt_exponential.h5
├── abrupt_gamma.h5
├── gradual_normal.h5
├── gradual_exponential.h5
├── gradual_gamma.h5
├── incremental_normal.h5
├── incremental_exponential.h5
└── incremental_gamma.h5
```

Each archive stores 1,000 streams in `values`, with shape `(1000, 70000)`, together with seeds, drift limits, transition lengths, and distribution parameters.

A reduced generation check is available through:

```bash
python scripts/generate_datasets.py --replications 2 --output-dir data/test
```

See `data/README.md` for additional details.

## Dataset-generation notebook

Open:

```text
notebooks/dataset_generation_demo.ipynb
```

Select the `MCDD` Conda environment as the kernel and run all cells. The notebook generates and validates one example for every drift/distribution combination and uses the publication-oriented plotting helper from the original experimentation workflow.

Figures are written to:

```text
notebooks/figures/dataset_generation/
```

## MCDD implementation

The detector is available as:

```python
from mcdd.detectors import MCDD
```

Example:

```python
from scipy.stats import ks_2samp
from mcdd.detectors import MCDD

detector = MCDD(
    ks_2samp,
    window_size=6_000,
    n_subwindows=10,
    alpha=0.01,
    window_mode="sliding",
    correction="fdr_by",
    min_rejections=1,
)
```

Supported window modes are:

- `sliding`: fixed-size sliding window;
- `growing_dynamic`: growing window with a fixed number of subwindows;
- `growing_fixed`: growing window with a fixed subwindow size and an increasing number of contrasts.

For `growing_fixed`, only complete subwindows are evaluated. When the maximum window size is not divisible by the subwindow size, the **most recent** complete portion is retained. For example, the 20,000-sample configuration uses the most recent 19,800 observations with 600-sample subwindows.

### Latched alarm state

MCDD keeps `drift_detected=True` after its first alarm. Later calls to `update` do not automatically return the detector to the non-drift state. Call:

```python
detector.reset()
```

before using the same instance to detect another drift. `reset()` clears the window, history, and latched alarm state.

## Experiment configurations

The experiment workflow evaluates ten configurations:

| Name | Method | Window strategy |
|---|---|---|
| `MCDD-S` | MCDD | Sliding window, 6,000 observations |
| `MCDD-G20k` | MCDD | Growing 6,000 → 20,000; fixed number of subwindows |
| `MCDD-G30k` | MCDD | Growing 6,000 → 30,000; fixed number of subwindows |
| `MCDD-G20kT` | MCDD | Growing 6,000 → 20,000; fixed 600-sample subwindows |
| `MCDD-G30kT` | MCDD | Growing 6,000 → 30,000; fixed 600-sample subwindows |
| `TSH-S` | Single KS test | Sliding window, 6,000 observations |
| `TSH-G20k` | Single KS test | Growing 6,000 → 20,000 |
| `TSH-G30k` | Single KS test | Growing 6,000 → 30,000 |
| `KSWIN` | River KSWIN | `window_size=6000`, `stat_size=600` |
| `LORD-LD` | LORD | Sliding window, 6,000 observations; local-dependence lag |

All configurations use `alpha=0.01`. MCDD uses the Benjamini--Yekutieli correction by default.

Only the locally dependent LORD procedure required by the experiment is kept. The unused asynchronous and classic implementations are not included.

## Experiment scoring

The evaluation preserves the convention used in the original experiments:

- valid detections use strict comparisons: `drift_start < detection < valid_detection_end`;
- abrupt drift has a 2,000-observation valid detection interval;
- gradual and incremental drift use their generated transition interval;
- when the first alarm is outside the valid interval, it is counted as a false alarm and the drift is not additionally counted as missed;
- when no alarm occurs, the drift is counted as missed.

The reported metrics are:

```text
FDR = FP / (TP + FP)
MDR = FN / (TP + FN)
IR  = TP / (TP + FP)
```

Mean delay is calculated only from valid detections.

## Run one detector on one dataset

Open:

```text
notebooks/run_experiments.ipynb
```

In **Run one detector on one dataset archive**, configure for example:

```python
RUN_SELECTED_EXPERIMENT = True
SELECTED_DATASET = "abrupt_normal.h5"
SELECTED_CONFIGURATION = "MCDD-S"
SELECTED_MAX_STREAMS = None
OVERWRITE_SELECTED_RESULTS = False
```

With `SELECTED_MAX_STREAMS = None`, all 1,000 streams in the selected archive are evaluated with that detector configuration. A small validation can be performed first by setting `SELECTED_MAX_STREAMS = 2`.

Focused outputs are written under `results/selected/`.

## Run the complete experiments

Open `notebooks/run_experiments.ipynb`, select the `MCDD` Conda environment, run the setup and validation cells, and then set:

```python
RUN_FULL_EXPERIMENTS = True
MAX_STREAMS_PER_ARCHIVE = None
```

The complete experiment evaluates 9,000 streams with 10 configurations, for a total of 90,000 detector-stream runs. It may require substantial execution time.

The notebook does not include SEED and does not include the former *Comparison: Sliding vs Growing Window* section.

## Results

The experiment notebook generates:

```text
results/
├── per_run_results.csv
└── summary_results.csv
```

At the current stage of the project, **generated CSV result files are intentionally not committed to the repository**. They are excluded through `.gitignore` and can be reproduced with the supplied workflow.

`per_run_results.csv` contains one row for every stream/configuration pair. `summary_results.csv` contains the corresponding counts and metrics. The final article-level averaging across distributions is performed separately and is not part of this repository.

See `results/README.md` for details.

## Automated tests

The `tests/` directory contains lightweight software checks. They do not reproduce the scientific benchmark; they verify that core implementation behavior remains stable after code changes.

Run the complete test suite from the repository root with:

```bash
python -m pytest -q
```

The tests cover deterministic generation, drift metadata, non-negative support for exponential and gamma data, MCDD alarm/reset behavior, `growing_fixed` window handling, strict detection scoring, HDF5 reading, and CSV generation.

## Citation

Citation metadata are provided in `CITATION.cff`. GitHub can use this file to expose a **Cite this repository** entry.

The preferred article citation is:

> Fernando Fraile Mulas, David Mera, and Rosa M. Crujeiras. *Multiple Contrast Drift Detection: False Discovery Rate Control for Concept Drift in Data Streams*. 2026.

A stable software release will be created once the final experiment results have been validated.

## License

The source code is released under the BSD 3-Clause License. See `LICENSE`.
