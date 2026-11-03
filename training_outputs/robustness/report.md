# DigitalTwin — frozen-model robustness evaluation

32 complete simulated shifts per condition. Model and threshold (0.94) remain fixed. No retraining.

| Condition | F1 | Recall | False alarms | Misses |
|---|---:|---:|---:|---:|
| Fresh simulated shifts | 0.929 | 0.896 | 23 | 71 |
| Additional sensor outages | 0.768 | 0.666 | 46 | 228 |
| Noisy measurements | 0.916 | 0.890 | 36 | 75 |
| Weaker disturbances | 0.928 | 0.920 | 37 | 46 |
| Concurrent disturbances | 0.935 | 0.917 | 86 | 159 |
| Healthy shifts | — | — | 12 | 0 |

- Fresh simulated shifts: Original data recipe, new seeds.
- Additional sensor outages: 20% of otherwise observable sensors unavailable for the full shift; paired with reference.
- Noisy measurements: Independent Gaussian measurement noise: standard deviation 20% of frozen training feature scales; paired with reference.
- Weaker disturbances: Fault severity sampled from 0.32–0.55.
- Concurrent disturbances: 4–6 injected faults in each non-healthy shift.
- Healthy shifts: No injected disturbances; evaluate false alarms.

- Same simulator, fault categories and fixed graph as training. No real-factory validation.
- Station-time observations are correlated; counts are not independent trials.
- F1, recall and average precision are undefined for all-healthy shifts.
- These challenge results are now inspected; do not tune against them as a final holdout.
