# 35°C charge-focus subset (SOC 20→80 with rests)

Source batch: longrun_cycling_35c_2026-05-30_19-47-06
Window: 2026-05-31 07:07:49 → 2026-05-31 21:07:51  (14.0 h)
Total scans: 93

## Segment breakdown

| step | tag    | SOC plateau | n scans | duration |
|---|---|---|---:|---|
| 5 | rest | 20 | 13 | 120 min |
| 6 | charge | charge | 14 | 120 min |
| 7 | rest | 40 | 13 | 120 min |
| 8 | charge | charge | 13 | 120 min |
| 9 | rest | 60 | 13 | 120 min |
| 10 | charge | charge | 14 | 120 min |
| 11 | rest | 80 | 13 | 120 min |

## Temperature stability
- TC08 line-mean T range: 35.803 → 35.957 °C
- T span: 154.2 mC
- T std: 33.8 mC


# 35°C charge-focus analyses (SOC 20→80 + rests)

## Within-rest evolution per plateau

All rests are ~117 min, 13 scans. End-of-rest drift vs σ_ROI noise floor.

### amplitude  (σ_ROI = 2.22 mV)

| SOC | ΔV (mV) | ΔT (mC) | cell raw end | ref mean end | cell − ref end | end-z (σ_ROI) | rate late/early |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | -11.0 | -63.3 | +2.99 | -24.2 | +27.1 | +12.2 | 5.03 |
| 40% | -6.0 | +29.4 | -28.6 | +24.4 | -53.1 | -23.9 | 0.61 |
| 60% | -12.0 | +5.4 | -9.19 | -28.8 | +19.6 | +8.8 | 0.18 |
| 80% | -7.0 | -10.4 | +6.39 | -8.06 | +14.5 | +6.5 | 1.32 |

### tof  (σ_ROI = 4.7 ns)

| SOC | ΔV (mV) | ΔT (mC) | cell raw end | ref mean end | cell − ref end | end-z (σ_ROI) | rate late/early |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | -11.0 | -63.3 | +30.1 | -57.4 | +87.5 | +18.6 | 5.65 |
| 40% | -6.0 | +29.4 | -55.2 | +171 | -227 | -48.2 | 0.86 |
| 60% | -12.0 | +5.4 | -11.1 | -101 | +90 | +19.1 | 0.30 |
| 80% | -7.0 | -10.4 | +39.2 | -26.9 | +66.1 | +14.1 | 1.59 |

### energy  (σ_ROI = 0.0814 a.u.)

| SOC | ΔV (mV) | ΔT (mC) | cell raw end | ref mean end | cell − ref end | end-z (σ_ROI) | rate late/early |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20% | -11.0 | -63.3 | +0.316 | -1.25 | +1.56 | +19.2 | 6.48 |
| 40% | -6.0 | +29.4 | -0.988 | +0.682 | -1.67 | -20.5 | 0.62 |
| 60% | -12.0 | +5.4 | -0.289 | -1.04 | +0.751 | +9.2 | 0.09 |
| 80% | -7.0 | -10.4 | +0.246 | -0.482 | +0.728 | +8.9 | 2.58 |


## Equilibrium amplitude: slope = -1.3869 mV/%SoC, max residual = 21.8 σ_ROI

## Equilibrium tof: slope = -3.1254 ns/%SoC, max residual = 6.0 σ_ROI

## Equilibrium energy: slope = -0.0528 a.u./%SoC, max residual = 15.5 σ_ROI
