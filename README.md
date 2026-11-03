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
- There are no missing readings in the supplied dataset. If a cycle contains invalid or incomplete features or signal samples, the API returns standard JSON `null` for unavailable readings and withholds all condition estimates. The dashboard displays missing signal samples as gaps and explains which channels are incomplete.

For each physical sensor, features are the mean, standard deviation, minimum, maximum and six consecutive ten-second means: **140 features per cycle**. Preparation reads one raw row at a time and writes a compact cache in `data/processed/`. This avoids loading the 531 MB of text into memory as one large matrix.

## What the dashboard provides

- **Overview:** four documented component-condition estimates, a review list, selectable measured signal and cycle statistics.
- **Sensors:** the 14 physical input channels, native rates and units, search/filter controls, and recorded traces.
- **Model validation:** component metrics, confusion matrices, linear/neural comparison, sensor-removal sensitivity and separately reported accumulator and GRU experiments.
- **Additional model research:** the validation page also shows the grouped boosted-tree comparison separately from the original deployed-model test.
- **Dataset & method:** source attribution, preprocessing, split protocol and limitations.
- **Cycle replay:** next, play/pause, reset, seek and a choice of test/validation/training/all recorded cycles. Cycle numbers refer to source-row indices; absolute recording timestamps are not supplied or invented.
- **Reference labels:** optional, explicitly requested annotations, shown separately from estimates and excluded from inference.
- **Exports:** current sensor statistics and predictions as CSV; full saved evaluation as JSON.

Each browser tab keeps its own record selection and sends it with requests. Changing replay in one tab cannot change another tab's displayed record, reference labels or CSV export. Reloading a tab starts at the first held-out record.

API clients select a record explicitly, for example `/api/state?partition=test&position=7`. Reference labels and CSV exports accept the same query parameters. Replay POST requests include `partition` and `position` in their JSON body; `/api/step` advances from that supplied position. Requests without a selection start from the default test partition at position zero.

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

## Accumulator upgrade experiment

```bash
.venv/bin/python -m scripts.compare_accumulator
```

This optional, CPU-only experiment saves results under `training_outputs/accumulator_experiment/` and **never replaces the deployed model**. It compares a freshly fitted linear baseline, a small neural network, and a linear model that uses nearest measured training cycles for the accumulator estimate. No extra package or generated sensor data is required.

The protocol was saved before fitting: seed 20261103 excludes the 29 previously inspected test combinations, then reserves 69 / 23 / 23 groups from the remaining combinations. These yield **1,164 / 477 / 236 training / validation / test cycles**. Normalization and neighbor references use training only. Nine predeclared neighbor candidates (3 / 7 / 15 neighbors, using pressure / pressure-power-flow / all physical features) are selected using validation only. Test scores are calculated afterward. The exact plan and indices are saved in `upgrade_protocol.json`; changing that frozen plan causes an error.

The candidate did **not** improve validation performance: accumulator macro F1 was **0.207** for the selected nearest-cycle candidate versus **0.674** for its freshly trained linear baseline. On this experiment's reserved test, the corresponding scores were **0.254 / 0.489**. The existing deployed model is retained. These experimental results use different test records from the original 328-cycle evaluation, so their scores are not a before/after comparison with its 0.869 accumulator F1.

See [the experiment report](training_outputs/accumulator_experiment/training_report.md). This is a re-reservation within previously explored, same-rig data; it is not independent-machine evidence. Re-running reproduces an inspected experiment and does not create a fresh test. Further tuning needs another declared evaluation protocol or genuinely independent measurements.

## GRU layer comparison

```bash
.venv/bin/python -m scripts.compare_gru
```

This completed CPU experiment compares the current linear architecture, fitted from scratch, with a **one-layer, 32-unit GRU**. The GRU receives 60 ordered time points × 14 physical sensor channels from the existing one-second averages. It resets its hidden state for every cycle. Its 5,070 parameters include a final output layer for the four conditions. Channel scaling and class weights use training cycles only.

The protocol was fixed before training. Seed 20261104 excludes both previously inspected test sets, then assigns the remaining 92 joint condition combinations to **55 / 18 / 19 training / validation / test groups**, or **804 / 424 / 413 cycles**. Each candidate uses the same cycles. The linear model uses its original 140 native-sample summary features; the GRU uses the averaged signal sequences, so their input representations differ.

| Model in this experiment | Parameters | Validation macro F1 | Test macro F1 | All four correct on test |
|---|---:|---:|---:|---:|
| Linear architecture, refitted | 1,974 | **0.8012** | **0.7029** | **35.84%** |
| GRU, one layer × 32 units | 5,070 | 0.5490 | 0.6483 | 14.29% |

The predeclared rule permits a two-layer GRU only if one layer beats the refitted linear model on validation mean component macro F1. **That condition was not met, so the second layer was not trained.** Checkpoints and the layer decision were saved before test scoring. The one-layer GRU trained in about 12 seconds on this machine; runtime varies.

All files stay in `training_outputs/gru_experiment/`. [The comparison report](training_outputs/gru_experiment/report.md), `protocol.json`, `selection.json` and exact test predictions document the result. The active dashboard model and its original **0.950** test macro F1 remain unchanged. That original result uses a different 328-cycle test and is not directly comparable with this table. This new split is an internal comparison within previously explored, same-rig data, not independent-machine evidence. One seed and one split do not measure variability across runs. Re-running reproduces an inspected experiment; changing the frozen settings causes an error.

## Boosted-tree comparison

This optional, CPU-only experiment tests a fixed, compact histogram gradient-boosted tree model against the **same linear architecture used by the dashboard**, refitted from scratch. Both use the 140 summary features computed from measured cycles. The tree comparison needs scikit-learn only when you run it; the dashboard and existing PyTorch models do not import that package.

```bash
.venv/bin/python -m pip install -r requirements-experiments.txt
.venv/bin/python -m scripts.compare_trees
```

The four grouped folds cover **1,228 development cycles / 73 condition combinations**. All 977 cycles in the three previously inspected test partitions are excluded. Within each outer fold, the linear model chooses its training epoch on a separate inner grouped validation, then refits on the full outer training fold. The boosted trees use one fixed setting and no random internal early-stopping split. Both models receive the same outer training records and are scored on the same validation records.

| Model | Mean across folds, macro F1 | Out-of-fold macro F1 | Out-of-fold accumulator F1 | All four correct |
|---|---:|---:|---:|---:|
| Linear architecture, refitted | **0.7719** | **0.8182** | **0.5154** | **47.64%** |
| Small boosted trees | 0.7433 | 0.7839 | 0.3662 | 29.72% |

Boosted trees trailed linear on **all four folds**. The original active model remains unchanged. The [report](training_outputs/tree_experiment/report.md), `protocol.json`, `metrics.json` and `out_of_fold_predictions.npz` preserve settings, partitions, fold scores and exact predictions. This is an internal comparison on the same rig. Its scores cannot be directly compared with the original deployed model's 0.950 score on a different test protocol, and it does not establish generalization to another machine.

## Sensor-removal sensitivity

```bash
.venv/bin/python -m scripts.evaluate_robustness
```

This offline diagnostic removes one physical channel at a time from the held-out measured features and replaces its feature block with **training means**. Model weights stay fixed. It is an artificial sensitivity experiment, not measurements from physical sensor failures, and it generates no synthetic raw signals.

The model policy is stricter: incomplete cycles produce `insufficient_data`; mean imputation is not silently used for displayed estimates. Invalid readings are exported as empty CSV cells and JSON `null`, and predictions remain unavailable for that cycle.

See [the sensitivity report](training_outputs/robustness/report.md) and [exact results](training_outputs/robustness/metrics.json). Now that these test results have been inspected, reserve a new independent evaluation protocol before future model tuning.

## Scope and limits

- The dataset comes from one physical test rig with correlated cycles. Generalization to another machine or a production factory has not been established.
- Classification uses the complete 60-second cycle. It does not establish advance warning, remaining useful life, causal diagnosis or intervention benefit.
- Softmax outputs are uncalibrated class support, not validated probabilities of physical failure.
- The component display is a monitoring summary. It does not claim a physical sensor-to-component topology absent from the source metadata.
- Original proposal documents in `papers/` are historical material, not evidence for this measured-data model.

Completed fixes, experiment findings and work requiring more data are recorded in [UPGRADES.md](UPGRADES.md). The current project remains a local, single-rig research application.

## Project structure

```text
app.py                         # Real-cycle dashboard and API
hydraulics.py                  # Measurement preparation, labels, splits and metrics
model.py                       # Small classifiers and experimental neighbor model
scripts/
  train_hydraulics.py           # Real-data training
  evaluate_robustness.py        # Frozen-model sensor-removal diagnostic
  compare_accumulator.py        # Separate, frozen accumulator experiment
  compare_gru.py                # Conditional one-/two-layer GRU comparison
  compare_trees.py              # Grouped CPU boosted-tree comparison
tests/
  test_hydraulics.py            # Provenance, splitting, normalization and reload
  test_gru.py                   # Sequence order, memory reset and comparison integrity
  test_trees.py                 # Group isolation and exact cross-validation scores
  test_monitoring.py            # API, abstention, replay and exports
  test_dashboard.cjs            # Missing-signal rendering and record selection
static/                        # Professional dashboard styling and interactions
templates/                     # Dashboard HTML
data/                          # Unchanged download and generated compact cache
training_outputs/              # Current REAL-data models and evidence
papers/                        # Original historical proposal documents
UPGRADES.md                    # Completed upgrades and remaining research needs
requirements-experiments.txt   # Optional tree-comparison dependency
```

## Checks

```bash
.venv/bin/python -B -m unittest discover -s tests -v
node --test tests/test_dashboard.cjs
```

Checks require the downloaded dataset and trained real-data artifacts. They verify raw-to-cache reproduction, condition decoding, split isolation, training-only normalization, checkpoint reload, exact metric reproduction, exclusion of labels from inference, strict JSON with invalid inputs, separate client selections, cycle navigation and matching exports. Experiment-specific checks run when the optional experiment artifacts exist. The JavaScript checks use Node's built-in test runner; the application itself still needs only Python. They check signal gaps, replay request selections and reference/export links without installing a browser.
