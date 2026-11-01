# RE-TEST b2048 — .so T3 prod (d2df07b8), MTP OFF

date : 2026-10-22 12:19 UTC
service restauré : OUI
.so prod : md5 prefix d2df07b8 (attendu d2df07b8)

A-b2048 = prod actuelle (T3 + -b 2048 -ub 2048) · B-b512 = T3 sans b2048
3 reps, ordre alterné, serveur frais par leg, port 8917, CVD=0, KV q8_0, FA auto, H2 OFF.

## DECODE (t/s) — le gain principal du 17/10 était ici
| contexte | rep | A b2048 | B b512 | Δ (A vs B) |
|---|---|---|---|---|
| 114688 | 1 | 21.03 | 20.97 | +0.3% |
| 114688 | 2 | 21.02 | 19.19 | +9.5% |
| 114688 | 3 | 21.02 | 18.87 | +11.4% |
| 196608 | 1 | 18.19 | 14.47 | +25.7% |
| 196608 | 2 | 18.18 | 14.38 | +26.4% |
| 196608 | 3 | 18.18 | 14.81 | +22.8% |

## DECODE — médianes par leg
| contexte | A b2048 | B b512 | Δ |
|---|---|---|---|
| 114688 | 21.02 | 19.19 | +9.5% |
| 196608 | 18.18 | 14.47 | +25.6% |

## PREFILL (t/s) — le levier -b
| contexte | A b2048 | B b512 | Δ |
|---|---|---|---|
| 114688 | 623.1 | 559.0 | +11.5% |
| 196608 | 465.7 | 426.6 | +9.2% |

