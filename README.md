# DigitalTwin — real hydraulic condition monitoring

A local condition-monitoring dashboard trained on the **downloaded UCI hydraulic test-rig measurements**. The active application, training commands and evaluation use real recorded sensor cycles. No simulator or synthetic assembly-line readings are used.

## Run

From this project directory, using the existing environment:

```bash
.venv/bin/python app.py
```

Open **http://127.0.0.1:5001**. The default replay partition is the held-out test set. Replay steps through historical records; it does not represent a live connection to equipment.

For a fresh environment, install `requirements.txt`, then train before starting the dashboard:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m scripts.train_hydraulics
.venv/bin/python -m scripts.evaluate_robustness
```

No GPU, graph-neural-network package, external service or LLM API is required. The deployed model and preprocessing are small enough for CPU use. The first complete training run took about 11 seconds on this machine; actual runtime varies.

## Dataset and attribution

**Condition monitoring of hydraulic systems**, Helwig, N., Pignanelli, E. & Schütze, A. (2015), UCI Machine Learning Repository. [Dataset and documentation](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems). DOI: **10.24432/C5CW21**. License: **CC BY 4.0**. Original measurements and annotations are preserved unchanged.

The downloaded files are under:

```text
data/condition+monitoring+of+hydraulic+systems/
```

Another location can be supplied through `HYDRAULICS_DATA_DIR`. Preparation automatically finds a complete dataset under `data/` and verifies raw-file hashes against its cache.

- **2,205 experimental cycles**, each 60 seconds, from one physical hydraulic test rig.
- **14 physical input channels:** PS1–PS6 pressure, EPS1 motor power, FS1–FS2 volume flow, TS1–TS4 temperature and VS1 vibration.
- Native sampling rates are 100 Hz, 10 Hz and 1 Hz. Charts show one-second averages; feature statistics use the original native samples.
- **CE, CP and SE**, supplied derived channels, are excluded from model inputs.
- `profile.txt` annotates cooler efficiency, valve condition, internal pump leakage, accumulator pressure condition and whether steady-state conditions were reached. These annotations are **never sensor input features**.
- There are no missing readings in the supplied dataset. The application abstains if incomplete/non-finite features are provided.

For each physical sensor, features are the mean, standard deviation, minimum, maximum and six consecutive ten-second means: **140 features per cycle**. Preparation reads one raw row at a time and writes a compact cache in `data/processed/`. This avoids loading the 531 MB of text into memory as one large matrix.

## What the dashboard provides

- **Overview:** four documented component-condition estimates, a review list, selectable measured signal and cycle statistics.
- **Sensors:** the 14 physical input channels, native rates and units, search/filter controls, and recorded traces.
- **Model validation:** component metrics, confusion matrices, linear/neural comparison and sensor-removal sensitivity.
- **Dataset & method:** source attribution, preprocessing, split protocol and limitations.
- **Cycle replay:** next, play/pause, reset, seek and a choice of test/validation/training/all recorded cycles. Cycle numbers refer to source-row indices; absolute recording timestamps are not supplied or invented.
- **Reference labels:** optional, explicitly requested annotations, shown separately from estimates and excluded from inference.
- **Exports:** current sensor statistics and predictions as CSV; full saved evaluation as JSON.

The previous vehicle assembly graph, queues, simulated defects, invented station labels and simulation controls are removed from the active project. The interface reflects the hydraulic dataset instead of forcing it into the old 40-station layout.

## Training and evaluation

```bash
.venv/bin/python -m scripts.train_hydraulics
```

All repeated cycles with the same joint four-component condition combination remain in **one** partition. Seed 42 assigns the 144 combinations to 86 training, 29 validation and 29 test groups. This produces **1,371 / 506 / 328 cycles**, respectively. The full set of cycle indices and combinations is saved in `training_outputs/split_manifest.json`.

This is a **same-rig held-out condition-combination experiment**. It is not chronological future prediction or independent-machine validation. Group definitions use annotations solely to build the evaluation partition, not as model inputs. The dataset does not provide independent experimental-run identifiers.

Feature scaling and class weights are fit only on training cycles. Both a class-weighted linear classifier and a small neural network are trained. The checkpoint with the highest validation mean component macro F1 is selected; test results are calculated after that choice.

The **linear classifier** was selected on validation. It has **1,974 parameters** and supplies the dashboard estimates.

| Held-out test result | Cooler | Valve | Pump | Accumulator | Mean macro F1 |
|---|---:|---:|---:|---:|---:|
| Linear — deployed | 1.000 | 0.935 | 0.996 | 0.869 | **0.950** |
| Small neural network | 1.000 | 0.915 | 0.988 | 0.701 | 0.901 |

For the deployed model, mean component accuracy is **94.97%**; all four condition estimates are simultaneously correct on **80.49%** of held-out cycles. These are different metrics. The accumulator is the weakest of the four tasks.

Artifacts:

- `model.pt`, `linear.pt`, `mlp.pt`: saved weights and training-only normalization; checkpoints are ignored by Git.
- [metrics.json](training_outputs/metrics.json): exact metrics, confusion counts, validation scores, epoch histories and artifact fingerprint.
- [training_report.md](training_outputs/training_report.md): readable results.
- [source_manifest.json](training_outputs/source_manifest.json): hashes and sizes of the downloaded text files.
- `test_predictions.npz`: original held-out annotations, cycle indices and exact evaluated predictions.

Restart the dashboard after retraining. It checks that the loaded model, cached data and reported metrics refer to the same source data and checkpoint. Without trained weights it shows measured readings with estimates unavailable; it never generates a random fallback model.

## Sensor-removal sensitivity

```bash
.venv/bin/python -m scripts.evaluate_robustness
```

This offline diagnostic removes one physical channel at a time from the held-out measured features and replaces its feature block with **training means**. Model weights stay fixed. It is an artificial sensitivity experiment, not measurements from physical sensor failures, and it generates no synthetic raw signals.

The production policy is stricter: incomplete features produce `insufficient_data`; mean imputation is not silently used for displayed estimates.

See [the sensitivity report](training_outputs/robustness/report.md) and [exact results](training_outputs/robustness/metrics.json). Now that these test results have been inspected, reserve a new independent evaluation protocol before future model tuning.

## Scope and limits

- The dataset comes from one physical test rig with correlated cycles. Generalization to another machine or a production factory has not been established.
- Classification uses the complete 60-second cycle. It does not establish advance warning, remaining useful life, causal diagnosis or intervention benefit.
- Softmax outputs are uncalibrated class support, not validated probabilities of physical failure.
- The component display is a monitoring summary. It does not claim a physical sensor-to-component topology absent from the source metadata.
- Original proposal documents and archived synthetic experiments in `papers/` are historical material, not evidence for this measured-data model.

## Project structure

```text
app.py                         # Real-cycle dashboard and API
hydraulics.py                  # Measurement preparation, labels, splits and metrics
model.py                       # Small linear / neural classifiers
scripts/
  train_hydraulics.py           # Real-data training
  evaluate_robustness.py        # Frozen-model sensor-removal diagnostic
tests/
  test_hydraulics.py            # Provenance, splitting, normalization and reload
  test_monitoring.py            # API, abstention, replay and exports
static/                        # Professional dashboard styling and interactions
templates/                     # Dashboard HTML
data/                          # Unchanged download and generated compact cache
training_outputs/              # Current REAL-data models and evidence
papers/                        # Historical proposal documents / synthetic archive
```

## Checks

```bash
.venv/bin/python -B -m unittest discover -s tests -v
```

Checks require the downloaded dataset and trained real-data artifacts. They verify raw-to-cache reproduction, condition decoding, split isolation, training-only normalization, checkpoint reload, exact metric reproduction, exclusion of labels from inference, abstention, cycle navigation and exports.
