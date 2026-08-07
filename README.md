# Multiple Contrast Drift Detection (MCDD)

This repository contains the synthetic data generators, detector implementation, experiment workflow, and reporting utilities used to evaluate **Multiple Contrast Drift Detection (MCDD)**.

The repository includes:

- generators for abrupt, gradual, and incremental concept drift;
- normal, exponential, and gamma data streams;
- reproducible HDF5 generation for the complete 9,000-stream benchmark;
- the official `MCDD` detector implementation;
- Traditional Single Hypothesis, KSWIN, and LORD-LD baselines;
- command-line and notebook workflows for complete and focused experiments;
- article-style result aggregation and LaTeX output;
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
│   ├── generate_datasets.py
│   ├── run_experiments.py
│   └── show_results.py
├── src/
│   └── mcdd/
│       ├── datasets/
│       │   └── generators.py
│       ├── detectors/
│       │   ├── mcdd.py
│       │   └── lord.py
│       └── experiments/
│           ├── evaluation.py
│           └── reporting.py
└── tests/
    ├── conftest.py
    ├── test_generators.py
    ├── test_mcdd.py
    ├── test_evaluation.py
    └── test_reporting.py
```

## Environment

Create the Conda environment used for the experiments with:

```bash
conda env create -f environment.yml
conda activate MCDD
```

The supplied environment was exported from a Windows x86-64 setup and therefore contains some platform-specific Conda packages. The machine-specific prefix was removed. `h5py` is included for the benchmark archives and `pytest` for the automated tests.

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

Gradual and incremental transitions are generated from the selected distribution. Exponential and gamma transition intervals are therefore not generated from a normal distribution.

### Generate the HDF5 files

The complete `.h5` benchmark files are intentionally not stored in GitHub because of their size. They can be reproduced from the fixed generation code and seeds provided here.

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

Each archive stores 1,000 streams in `values`, with shape `(1000, 70000)`, together with seeds, drift limits, transition lengths, and distribution parameters.

A reduced generation check can be run with:

```bash
python scripts/generate_datasets.py --replications 2 --output-dir data/test
```

See `data/README.md` for additional details.

## Dataset-generation notebook

Open:

```text
notebooks/dataset_generation_demo.ipynb
```

Select the `MCDD` Conda environment and run all cells. The notebook generates and validates one example for every drift/distribution combination.

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

For `growing_fixed`, only complete subwindows are evaluated. When the maximum window size is not divisible by the subwindow size, the most recent complete portion is retained. For example, the 20,000-sample configuration uses the most recent 19,800 observations with 600-sample subwindows.

MCDD uses a latched alarm state: after the first alarm, `drift_detected` remains `True` until:

```python
detector.reset()
```

is called.

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

## Experiment scoring

The evaluation preserves the original experiment convention:

- valid detections use the strict condition `drift_start < detection < valid_detection_end`;
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

Mean Delay is calculated only from valid detections.

## Run the complete experiments from the terminal

After generating the nine HDF5 files, run:

```bash
python scripts/run_experiments.py
```

This evaluates all 10 detector configurations on all 9,000 streams, corresponding to 90,000 detector-stream runs, and generates:

```text
results/
├── per_run_results.csv
└── summary_results.csv
```

For a reduced validation using two streams from each archive:

```bash
python scripts/run_experiments.py --max-streams 2
```

To replace existing results explicitly:

```bash
python scripts/run_experiments.py --overwrite
```

Use:

```bash
python scripts/run_experiments.py --help
```

for all options.

## Notebook experiment workflow

The same workflow is available interactively in:

```text
notebooks/run_experiments.ipynb
```

The notebook includes:

- dataset validation;
- a quick one-stream check;
- execution of one detector on one selected dataset;
- complete 90,000-run execution;
- inspection of generated CSV files;
- article-style result tables;
- optional LaTeX table generation.

For the complete benchmark set:

```python
RUN_FULL_EXPERIMENTS = True
MAX_STREAMS_PER_ARCHIVE = None
```

## Article-style result tables

The article tables are generated from the distribution-specific rows in `results/summary_results.csv`.

For every detector and drift type, the reporting layer computes the arithmetic mean of FDR, MDR, IR, and Mean Delay across:

```text
normal
exponential
gamma
```

The `distribution="all"` pooled rows created by `summarize_results` are deliberately ignored for this step.

The overall comparison is then obtained by averaging the three drift-level values:

```text
abrupt
gradual
incremental
```

For scenarios in which a detector produces only false alarms and no valid drift detections, the article-style `NA` convention is applied to metrics that are not meaningful.

### Display the article tables from the terminal

After `summary_results.csv` has been generated:

```bash
python scripts/show_results.py
```

This prints, with four decimal places:

```text
Abrupt drift
Gradual drift
Incremental drift
Overall comparison
```

To print LaTeX versions:

```bash
python scripts/show_results.py --latex
```

Other options include:

```bash
python scripts/show_results.py --decimals 4
python scripts/show_results.py --summary-file path/to/summary_results.csv
python scripts/show_results.py --help
```

The underlying functions are implemented in `src/mcdd/experiments/reporting.py` and are also used directly by the experiment notebook.

## Results and version control

Generated result CSV files are intentionally not committed to the repository at this stage. They are excluded through `.gitignore` and can be reproduced with the supplied workflows.

See `results/README.md` for details.

## Automated tests

Run the complete software test suite with:

```bash
python -m pytest -q
```

The tests cover deterministic generation, drift metadata, non-negative support for exponential and gamma data, MCDD alarm/reset behavior, `growing_fixed` window handling, strict detection scoring, HDF5 reading, CSV generation, and article-style result aggregation.

## Citation

Citation metadata are provided in `CITATION.cff`. GitHub can use this file to expose a **Cite this repository** entry.

Preferred article citation:

> Fernando Fraile Mulas, David Mera, and Rosa M. Crujeiras. *Multiple Contrast Drift Detection: False Discovery Rate Control for Concept Drift in Data Streams*. 2026.

A stable software release will be created once the final experiment results have been validated.

## License

The source code is released under the BSD 3-Clause License. See `LICENSE`.
