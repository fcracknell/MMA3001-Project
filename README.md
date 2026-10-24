# SGP4 Error Correction with Gradient Boosting

**MMA3001 Numerical Methods and Machine Learning, individual project.** Finn Cracknell, October 2026.

SGP4 is the standard propagator for the public satellite catalogue (two-line element sets, TLEs). Its predictions drift by kilometres over a few days, mainly because atmospheric drag is mis-estimated in low Earth orbit (LEO). This project trains a model to predict SGP4's error from information available when an element set is issued, and subtracts it.

**Result.** On 39 debris objects and a three-month period never seen in training, gradient boosting reduced RMSE position error by **12.5% (95% CI 4.4–17.1%)** relative to uncorrected SGP4. The gain comes from the largest, drag-driven errors below 700 km; typical (median) errors were not significantly changed. A physics-based ridge regression gave no improvement. Ground truth is a later-issued element set, so results measure consistency between element sets, not true position error. Full details are in the report (`report/`).

## Repository layout

```
MMA3001-Project/
├── src/sgp4corr/           # Source code (one module per pipeline stage)
│   ├── download.py         #   1. select objects, download element sets and space weather
│   ├── frames.py           #      RTN (radial, along-track, cross-track) frame transformation
│   ├── pairs.py            #   2. propagate element sets and compute SGP4 residuals
│   ├── features.py         #   3. build model inputs, flag outliers
│   ├── models.py           #   4. split data, train ridge and gradient boosting
│   ├── evaluate.py         #      test-set metrics, bootstrap CIs, figures
│   └── performance.py      #   5. timing, profiling, feature importance
├── tests/                  # pytest test suite (32 tests)
├── data/
│   ├── candidates.csv      # Objects sampled from the satellite catalogue
│   ├── objects.csv         # Final 246 objects used (committed for reproducibility)
│   ├── raw/                # Downloaded data (git-ignored, see "Data" below)
│   └── processed/          # Pairs, features and predictions (git-ignored, regenerated)
├── models/                 # Trained models (git-ignored, regenerated)
├── reports/                # Results: CSV tables, run outputs, profiles
│   └── figures/            # Figures used in the report
├── report/                 # Report source (LaTeX) and PDF
├── docs/                   # HTML documentation of the source code (pdoc)
├── requirements.txt
└── LICENSE                 # GNU GPL v3
```

## Setup

Tested with **Python 3.14** on Windows. Any Python 3.10 or later should work.

```powershell
git clone <repository-url>
cd MMA3001-Project
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Space-Track account

The element-set history comes from [Space-Track.org](https://www.space-track.org), which requires a free account. Credentials are read from environment variables and are never stored in the repository:

```powershell
# Windows PowerShell
$env:SPACETRACK_USER="you@example.com"
$env:SPACETRACK_PASS="your-password"
```

```bash
# macOS / Linux
export SPACETRACK_USER="you@example.com"
export SPACETRACK_PASS="your-password"
```

These only last for the current terminal session.

## Reproducing the results

Run each stage from the repository root, in order. Each stage caches its output, so a later stage can be re-run without repeating earlier ones.

| Step | Command | Output | Time (approx.) |
|---|---|---|---|
| 1. Download data | `python src/sgp4corr/download.py` | `data/raw/`, `data/objects.csv` | 10–15 min (rate-limited) |
| 2. Build prediction pairs | `python src/sgp4corr/pairs.py` | `data/processed/pairs.parquet`, checkpoint figure | 5 min |
| 3. Build features | `python src/sgp4corr/features.py` | `data/processed/features.parquet` | 2–3 min |
| 4. Train and evaluate | `python src/sgp4corr/models.py` | `models/`, `reports/results_*.csv`, figures | 4–5 min |
| 5. Performance analysis | `python src/sgp4corr/performance.py` | `reports/performance_*`, `profile_*`, feature importance | 2 min |

`python src/sgp4corr/evaluate.py` re-runs the evaluation alone from saved predictions. Every script accepts `--help` for its options (data window, number of hyperparameter candidates, random seed, and so on).

**Reproducibility notes.**
- All random choices (object sampling, pair selection, data split, model training) use fixed seeds.
- `download.py` reuses `data/candidates.csv` if it exists, so re-running it fetches the same objects even after the live catalogue changes.
- Results may differ slightly from those in `reports/` if Space-Track has re-issued element sets for the data window since October 2026.

## Data

| Source | What | Licence / terms |
|---|---|---|
| [Space-Track.org](https://www.space-track.org) (US Space Force) | Element-set history (`gp_history`), 1 July 2024 to 1 July 2026 | Free account; the user agreement restricts redistribution, so raw data is **not** included. Run step 1 to download it. |
| [CelesTrak SATCAT](https://celestrak.org/pub/satcat.csv) | Satellite catalogue used to select objects | Public |
| [CelesTrak space weather](https://celestrak.org/SpaceData/SW-All.csv) | Daily F10.7 solar flux and Ap geomagnetic index | Public |

The final object list (`data/objects.csv`: 246 debris objects and rocket bodies, perigee 300–2000 km) is committed, since it is a list of catalogue numbers rather than Space-Track data.

## Inputs and outputs

**Input:** an element set (mean orbital elements, as published by Space-Track) and a prediction horizon of 0–7 days. **Output:** SGP4's predicted position error in km, in radial, along-track and cross-track components; the corrected position is the SGP4 position minus this error. The report (Section 2) lists every model input with units and domain. The supported domain is non-manoeuvring objects with perigee 300–2000 km and horizons up to 7 days; inputs outside it are not rejected automatically.

## Tests

```powershell
python -m pytest                                                      # run all 32 tests
python -m pytest --html=reports/test_report.html --self-contained-html # HTML test report
```

The tests cover the RTN frame transformation, element-set loading (checked against the reference TLE parser), residual computation, absence of data leakage in features and in the train/validation/test split, the physical constraints on the models, and equivalence of the optimised and original code.

## Documentation

HTML documentation of every module, generated from the docstrings (Google style), is in `docs/`. Open `docs/index.html` in a browser. To regenerate it:

```powershell
pdoc src/sgp4corr -o docs
```

## Development workflow

Development used Git with descriptive commits. The documentation and report stages were developed on feature branches (`docs/readme`, `docs/html`, `test/report`, `report/final`) and merged into `main` through pull requests. Tags mark the three model iterations described in the report:

| Tag | Model iteration |
|---|---|
| `run1-shared-validation` | Validation set shared the training period; leakage found |
| `run2-time-split` | Validation moved to its own time period |
| `run3-final` | Corrections constrained to vanish at zero horizon (final model) |

Saved outputs of each run are in `reports/run1_output.txt`, `run2_output.txt` and `run3_output.txt`.

## AI acknowledgement

Generative AI (Claude, by Anthropic, via the Claude app, October 2026) was used throughout this project:

- **Scoping and planning:** critiquing the initial proposal (in particular the choice of ground truth) and drafting a staged plan.
- **Code:** drafting the source modules and their tests.
- **Troubleshooting:** Git, GitHub and PowerShell problems.
- **Interpretation:** reading the results tables.
- **Writing:** drafting the report and this README.

All code was run, tested and checked by me against physical expectations and an uncorrected baseline, and I made the key decisions on scope, ground truth, model iterations and when to stop. Errors made by the AI, and how they were found, are discussed in the report's AI reflection (Section 8). I take full responsibility for all code, results and conclusions in this repository.

## Licence

GNU General Public License v3.0; see `LICENSE`.