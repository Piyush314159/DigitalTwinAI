# GRU comparison on measured hydraulic cycles

Same cycle partitions for every freshly fitted candidate: {'train': 804, 'validation': 424, 'test': 413}; condition groups: {'train': 55, 'validation': 18, 'test': 19}. Both previously inspected test sets are excluded.

| Model | Layers | Parameters | Validation macro F1 | Test macro F1 | Test accumulator F1 | All four correct | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| linear | — | 1,974 | 0.8012 | 0.7029 | 0.6181 | 35.84% | 1.4 |
| gru_1 | 1 | 5,070 | 0.5490 | 0.6483 | 0.3214 | 14.29% | 12.3 |

One-layer GRU did not improve validation; extra layers skipped as requested. Validation-selected experiment winner: **linear**.

The current dashboard model and its original 0.950 test score are unchanged. That score uses different records and cannot be directly ranked against this table.

Input: 60 ordered one-second averages × 14 measured channels. Hidden state resets per cycle. GRU normalizers and loss weights are fitted on training only. All layer choices and checkpoint selection use validation only; test is scored afterward.

- Re-reserved groups from previously explored same-rig measurements, not new independent-machine evidence. Old deployed 0.950 score uses different test cycles.
- Different input representations: native-sample summaries for linear; one-second averaged sequences for GRU.
- Full-cycle condition classification, not future failure prediction. Scores are uncalibrated class support.
- One training seed and one grouped split; this does not measure variation across runs.
