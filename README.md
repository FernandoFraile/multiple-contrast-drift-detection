# Multiple Contrast Drift Detection (MCDD)

This repository contains the synthetic data generators, detector implementation, experiment workflow, and reporting utilities used to evaluate **Multiple Contrast Drift Detection (MCDD)**.

The repository includes:

- generators for abrupt, gradual, and incremental concept drift;
- normal, exponential, and gamma data streams;
- reproducible HDF5 generation for the complete 90,000-stream benchmark;
- the `MCDD` detector implementation;
- Traditional Single Hypothesis, KSWIN, and LORD-LD baselines;
- command-line and notebook workflows for complete and focused experiments;
- article-style result aggregation and LaTeX output;
- automated software tests.


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
│   └── README.md
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

Gradual transitions mix observations from the old and new concepts according to a sigmoid probability. Incremental transitions interpolate the parameters of the selected distribution throughout the transition interval.

### Generate the HDF5 files

The complete `.h5` benchmark files are not stored in GitHub because of their size. They can be reproduced from the generation code and fixed seeds provided here.

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

MCDD uses a latched alarm state: after the first alarm, `drift_detected` remains `True` until `detector.reset()` is called.

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

Each stream contains one known concept drift. The first alarm is classified relative to the drift start and the valid detection deadline:

- `alarm_index < drift_start` → **false alarm (FP)**;
- `drift_start <= alarm_index <= valid_detection_end` → **valid detection (TP)**;
- `alarm_index > valid_detection_end` → **late detection (FN)**;
- no alarm → **missed detection (FN)**.

For abrupt drift, the valid detection deadline is 2,000 observations after the drift point. For gradual and incremental drift, the deadline is the end of the generated transition interval.

Late detections are retained as a distinct per-run outcome and their delay beyond the deadline is stored in `late_delay`, but they contribute to FN rather than FP. Thus each run satisfies:

```text
TP + FP + FN = 1
```

The reported metrics are:

```text
FDR = FP / (TP + FP)
MDR = FN / (TP + FN)
IR  = TP / (TP + FP)
```

If a denominator is zero, the corresponding metric is undefined and is stored as `NaN`. Mean Delay is calculated only from valid detections and is also `NaN` when no valid detection exists.

## Run experiments from the terminal

After generating the HDF5 files, the complete benchmark can be run with:

```bash
python scripts/run_experiments.py
```

This evaluates all 10 detector configurations on all 9,000 streams, corresponding to 90,000 detector-stream runs, and generates:

```text
results/
├── per_run_results.csv
└── summary_results.csv
```

### Partial execution by drift type

Run abrupt drift first:

```bash
python scripts/run_experiments.py --drift abrupt
```

Append gradual drift to the same master CSV files:

```bash
python scripts/run_experiments.py --drift gradual --append
```

Then append incremental drift:

```bash
python scripts/run_experiments.py --drift incremental --append
```

The `--append` mode checks the run key (`configuration`, `drift_type`, `distribution`, `row_index`) and refuses to add duplicate runs.

### Reduced validation

For example, test abrupt drift using only two streams from each distribution:

```bash
python scripts/run_experiments.py --drift abrupt --max-streams 2
```

Replace a reduced validation result before a definitive run with:

```bash
python scripts/run_experiments.py --drift abrupt --overwrite
```

### More specific filters

A single distribution can be selected:

```bash
python scripts/run_experiments.py --drift abrupt --distribution normal
```

A single detector configuration can also be selected:

```bash
python scripts/run_experiments.py --drift abrupt --configuration MCDD-S
```

Filters can be combined and repeated to select multiple values. Use `python scripts/run_experiments.py --help` for all options.

## Notebook experiment workflow

The same workflow is available interactively in:

```text
notebooks/run_experiments.ipynb
```

The notebook includes dataset validation, a quick one-stream check, selected experiments, complete experiment execution, inspection of generated CSV files, article-style result tables, and optional LaTeX output.

## Article-style result tables

The article tables are generated from the distribution-specific rows in `results/summary_results.csv`.

For every detector and drift type, the reporting layer computes the arithmetic mean of FDR, MDR, IR, and Mean Delay across normal, exponential, and gamma distributions. The `distribution="all"` pooled rows created by `summarize_results` are not used for this step.

The overall comparison is obtained by averaging the three drift-level values: abrupt, gradual, and incremental.

Display the tables with:

```bash
python scripts/show_results.py
```

To print LaTeX versions:

```bash
python scripts/show_results.py --latex
```

## Results and version control

Generated result CSV files are excluded through `.gitignore` and can be reproduced with the supplied workflows. See `results/README.md` for details.

## Automated tests

Run the complete software test suite with:

```bash
python -m pytest -q
```

The tests cover deterministic generation, drift metadata, non-negative support for exponential and gamma data, MCDD alarm/reset behavior, growing-window handling, detection scoring, HDF5 reading, CSV generation, metric aggregation, and article-style result reporting.

## Citation

Citation metadata are provided in `CITATION.cff`. GitHub can use this file to expose a **Cite this repository** entry.

Preferred article citation:

> Fernando Fraile Mulas, David Mera, and Rosa M. Crujeiras. *Multiple Contrast Drift Detection: False Discovery Rate Control for Concept Drift in Data Streams*. 2026.

A stable software release will be created once the final experiment results have been validated.

## License

The source code is released under the BSD 3-Clause License. See `LICENSE`.
