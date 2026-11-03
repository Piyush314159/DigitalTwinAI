# Boosted-tree comparison on measured hydraulic cycles

Grouped 4-fold comparison on 1228 development cycles from 73 condition combinations. The 977 cycles in three previously inspected test partitions were excluded.

| Model | Fold mean macro F1 | Out-of-fold macro F1 | Out-of-fold accumulator F1 | All four correct | Total fitting time |
|---|---:|---:|---:|---:|---:|
| Linear architecture, refitted | 0.7719 | 0.8182 | 0.5154 | 47.64% | 3.8s |
| Histogram gradient-boosted trees | 0.7433 | 0.7839 | 0.3662 | 29.72% | 58.5s |

Trees beat linear on 0 of 4 grouped folds. The active dashboard model is unchanged.

The linear model chooses its epoch on an inner grouped validation, then refits on the full outer training fold. The tree settings are fixed before evaluating any outer fold; there is no random internal early stopping split. Both models use the same 140 native-sample summary features.

- All records come from the same previously explored test rig. Grouped cross-validation is an internal estimate, not a fresh independent-machine test.
- Four grouped folds measure variation across these condition combinations, not variation across machines or independent runs.
- All input features summarize the complete 60-second cycle; these scores are condition classification, not future failure prediction.
- The original deployed 0.950 test score uses a different protocol and cannot be ranked directly against these cross-validation scores.
