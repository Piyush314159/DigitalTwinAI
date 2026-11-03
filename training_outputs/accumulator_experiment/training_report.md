# DigitalTwin — real hydraulic measurements

Source: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

Seed 20261103; exclude all 29 previously inspected test combinations. Split the remaining 115 combinations into 69 training, 23 validation and 23 new test groups. Fit all models from scratch.

Cycles: {'train': 1164, 'validation': 477, 'test': 236}. Experiment winner chosen on validation: **linear**.

| Component | MLP macro F1 | Linear macro F1 | Linear + neighbors macro F1 | Experiment winner accuracy |
|---|---:|---:|---:|---:|
| Cooler | 1.000 | 1.000 | 1.000 | 1.000 |
| Valve | 0.942 | 0.960 | 0.960 | 0.962 |
| Pump | 1.000 | 0.993 | 0.993 | 0.996 |
| Accumulator | 0.555 | 0.489 | 0.254 | 0.483 |

Accumulator neighbor search: 9 validation-only candidates. Selected 15 neighbors using all_physical features. References are training cycles only. The deployed model is unchanged by this experiment.

Experiment winner mean component macro F1: 0.860. Mean component accuracy: 86.02%. All four components correct: 46.19%.

- Same physical test rig; not an independent-machine or production-factory validation.
- Joint condition combinations are held out; this is not a chronological forecast.
- Uses a complete 60-second cycle. No advance warning or remaining useful life is established.
- Scores are uncalibrated class support (softmax or neighbor votes). No causal root cause or intervention benefits are established.
- Re-reserved cycles from the same previously explored dataset, not independent-machine or previously unseen external data.
- No run identifiers are supplied; cycles within this experiment can be correlated.
- Original records contain no missing values; sensor removal tests are controlled sensitivity experiments.

Decision: retain_existing_deployment. Experimental results never replace the active deployment. Compare models within this protocol; its test set differs from the original 328-cycle test.
