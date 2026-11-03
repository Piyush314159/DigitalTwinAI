# DigitalTwin — real hydraulic measurements

Source: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems

Seed 42; 60/20/20 split of all 144 joint condition combinations. Entire combinations, including repeated cycles, stay together.

Cycles: {'train': 1371, 'validation': 506, 'test': 328}. Deployed model chosen on validation: **linear**.

| Component | MLP macro F1 | Linear macro F1 | Deployed accuracy |
|---|---:|---:|---:|
| Cooler | 1.000 | 1.000 | 1.000 |
| Valve | 0.915 | 0.935 | 0.936 |
| Pump | 0.988 | 0.996 | 0.997 |
| Accumulator | 0.701 | 0.869 | 0.866 |

Deployed mean component macro F1: 0.950. Mean component accuracy: 94.97%. All four components correct: 80.49%.

- Same physical test rig; not an independent-machine or production-factory validation.
- Joint condition combinations are held out; this is not a chronological forecast.
- Uses a complete 60-second cycle. No advance warning or remaining useful life is established.
- Scores are uncalibrated class support. No causal root cause or intervention benefits are established.
- No run identifiers are supplied; cycles within this experiment can be correlated.
- Original records contain no missing values; sensor removal tests are controlled sensitivity experiments.
