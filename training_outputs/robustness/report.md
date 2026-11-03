# Real-data sensor-removal sensitivity

Frozen deployed classifier; one physical sensor removed at a time; its ten features replaced by training means. All targets are original UCI annotations.

| Condition | Mean component macro F1 | All four correct |
|---|---:|---:|
| Complete measurements | 0.950 | 0.805 |
| PS1 removed | 0.704 | 0.259 |
| PS2 removed | 0.780 | 0.320 |
| PS3 removed | 0.700 | 0.149 |
| PS4 removed | 0.909 | 0.683 |
| PS5 removed | 0.902 | 0.631 |
| PS6 removed | 0.891 | 0.582 |
| EPS1 removed | 0.762 | 0.299 |
| FS1 removed | 0.702 | 0.369 |
| FS2 removed | 0.786 | 0.290 |
| TS1 removed | 0.869 | 0.509 |
| TS2 removed | 0.874 | 0.527 |
| TS3 removed | 0.878 | 0.546 |
| TS4 removed | 0.878 | 0.546 |
| VS1 removed | 0.837 | 0.372 |

Incomplete cycles produce insufficient_data, not imputed normal estimates.
