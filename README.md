# Multiple Contrast Drift Detection (MCDD)

This repository contains the synthetic datasets and experiment implementation
used to evaluate **Multiple Contrast Drift Detection (MCDD)**.

The current version includes:

- generators for abrupt, gradual, and incremental drift;
- normal, exponential, and gamma data streams;
- HDF5 generation for the complete 9,000-stream benchmark;
- a visual dataset-generation notebook;
- the official `MCDD` detector implementation;
- Traditional Single Hypothesis, KSWIN, and LORD-LD experiment baselines;
- a notebook that evaluates all configurations on the generated HDF5 files.

SEED is not included in the current experiment implementation.

## Repository structure

```text
.
├── environment.yml
├── data/
│   └── datasets/
├── notebooks/
│   ├── dataset_generation_demo.ipynb
│   └── run_experiments.ipynb
├── results/
├── scripts/
│   └── generate_datasets.py
└── src/
    └── mcdd/
        ├── datasets/
        │   └── generators.py
        ├── detectors/
        │   ├── mcdd.py
        │   └── lord.py
        └── experiments/
            └── evaluation.py
```

## Environment

Create the Conda environment from the environment used in the original
experimentation:

```bash
conda env create -f environment.yml
conda activate MCDD
```

The machine-specific Conda prefix was removed, and `h5py` was added to support
the HDF5 benchmark files.

## Synthetic datasets

The complete benchmark contains:

- 3 drift types: abrupt, gradual, and incremental;
- 3 distributions: normal, exponential, and gamma;
- 1,000 replications for each combination;
- 70,000 observations per stream;
- drift starting at observation 40,000;
- additive Gaussian noise with standard deviation 0.1;
- transition lengths of 1,000, 2,000, or 3,000 observations for gradual and
  incremental drift.

Gradual and incremental transitions are generated from the selected
distribution. Exponential and gamma transition intervals are therefore not
generated from a normal distribution.

### Generate the HDF5 files

From the repository root:

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

Each archive stores 1,000 streams in `values`, with shape `(1000, 70000)`,
together with seeds, drift limits, transition lengths, and distribution
parameters.

A reduced generation test is available through:

```bash
python scripts/generate_datasets.py \
    --replications 2 \
    --output-dir data/test
```

## Dataset-generation notebook

Open:

```text
notebooks/dataset_generation_demo.ipynb
```

Select the `articulo` Conda environment as the kernel and run all cells. The
notebook generates and validates one example for each drift/distribution
combination and uses the publication-oriented `plot_dataset` helper from the
original experimentation notebook.

Figures are written to:

```text
notebooks/figures/dataset_generation/
```

## MCDD implementation

The official detector is available as:

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
- `growing_fixed`: growing window with a fixed subwindow size and an increasing
  number of contrasts.

For `growing_fixed`, only complete subwindows are evaluated. When the maximum
window size is not divisible by the subwindow size, the **most recent** complete
portion is retained. For example, the 20,000-sample configuration uses the most
recent 19,800 observations with 600-sample subwindows.

### Latched alarm state

MCDD keeps `drift_detected=True` after its first alarm. Later calls to `update`
do not automatically return the detector to the non-drift state. Call:

```python
detector.reset()
```

before using the same instance to detect another drift. `reset()` clears the
window, history, and latched alarm state.

## Experiment configurations

The notebook evaluates the following ten configurations:

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

All configurations use `alpha=0.01`. MCDD uses the Benjamini--Yekutieli
correction by default.

The unused asynchronous and classic LORD implementations are not included.
Only the locally dependent LORD procedure required by the experiment is kept.

## Experiment scoring

The evaluation deliberately preserves the convention used in the original
experiments:

- valid detections use strict comparisons:
  `drift_start < detection < valid_detection_end`;
- abrupt drift has a 2,000-observation valid detection interval;
- gradual and incremental drift use their generated transition interval;
- when the first alarm is outside the valid interval, it is counted as a false
  alarm and the drift is not additionally counted as missed;
- when no alarm occurs, the drift is counted as missed.

The reported metrics are:

```text
FDR = FP / (TP + FP)
MDR = FN / (TP + FN)
IR  = TP / (TP + FP)
```

Mean delay is calculated only from valid detections.


## Run one detector on one dataset

The experiment notebook also supports a focused execution consisting of one
detector configuration applied to every stream in one HDF5 archive.

Open:

```text
notebooks/run_experiments.ipynb
```

In the section **Run one detector on one dataset archive**, configure:

```python
RUN_SELECTED_EXPERIMENT = True

SELECTED_DATASET = "abrupt_normal.h5"
SELECTED_CONFIGURATION = "MCDD-S"

SELECTED_MAX_STREAMS = None
OVERWRITE_SELECTED_RESULTS = False
```

With `SELECTED_MAX_STREAMS = None`, this executes:

```text
1 selected HDF5 archive × 1 selected detector × 1,000 streams
= 1,000 detector–stream runs
```

Available detector configuration names are:

```text
MCDD-S
MCDD-G20k
MCDD-G30k
MCDD-G20kT
MCDD-G30kT
TSH-S
TSH-G20k
TSH-G30k
KSWIN
LORD-LD
```

A small validation can be performed first with:

```python
SELECTED_MAX_STREAMS = 2
```

Results are written under:

```text
results/selected/
├── abrupt_normal__mcdd_s_per_run.csv
└── abrupt_normal__mcdd_s_summary.csv
```

The per-run file contains one row per stream. The summary file contains one row
with the aggregated `TP`, `FP`, `FN`, `FDR`, `MDR`, `IR`, and mean detection
delay for the selected archive and detector.

## Run the experiments

Open:

```text
notebooks/run_experiments.ipynb
```

Then:

1. select the `articulo` Conda environment;
2. run the repository setup and dataset validation cells;
3. run the quick validation on the first abrupt-normal stream;
4. set `RUN_FULL_EXPERIMENTS = True`;
5. run the full experiment cell.

The complete experiment evaluates 9,000 streams with 10 configurations, for a
total of 90,000 detector–stream runs. It may require substantial execution
time.

The notebook does not include SEED and does not include the former
*Comparison: Sliding vs Growing Window* section.

## Results

The experiment notebook creates:

```text
results/
├── per_run_results.csv
└── summary_results.csv
```

`per_run_results.csv` contains one row for every stream and configuration.
`summary_results.csv` contains counts and metrics:

- for each drift type and distribution;
- aggregated across distributions for each drift type;
- aggregated across all evaluated scenarios.
