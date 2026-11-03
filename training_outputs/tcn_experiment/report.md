# Multirate TCN on measured hydraulic cycles

Three grouped outer folds across all 2,205 measured cycles. Every repeated joint condition combination stays in one fold. Both models use the same outer training and validation cycles. Training epochs are chosen on an inner grouped partition and refitted on the full outer training fold.

| Model | Parameters | Fold mean macro F1 | Out-of-fold macro F1 | Accumulator F1 | All four correct | Fit time |
|---|---:|---:|---:|---:|---:|---:|
| Linear, refitted | 1,974 | 0.8957 | 0.8959 | 0.7266 | 68.39% | 4.4s |
| Two-block multirate TCN | 12,558 | 0.9449 | 0.9453 | 0.8739 | 82.22% | 261.2s |

TCN beats linear on 3 of 3 folds. The deployed dashboard model remains unchanged.

Input: 9 physical high-rate channels at 10 Hz (pressure/power block averages and native flow), 5 native 1 Hz channels, and 140 measured-sample summary features. Two residual dilated temporal blocks, 16 high-rate channels, 8 low-rate channels and a 32-unit fusion head. No simulated signals or annotation features.

Post-hoc baseline sensitivity: refitting linear for 250 epochs on each outer training fold yielded fold-mean macro F1 0.9156. This check was added after the first outer-fold results were observed and is not a new held-out evaluation. Exact predictions are saved beside this report.

- All 2,205 cycles belong to the previously explored same-rig dataset. This is an internal exploratory comparison, not a new external test.
- Pressure and power were reduced from 100 Hz to 10 Hz by block averaging; high-frequency transients may be lost.
- The TCN consumes the complete 60-second cycle and cannot predict future failures or remaining useful life.
- The original deployed 0.950 score uses a different split and cannot be directly compared with these cross-validation values.
