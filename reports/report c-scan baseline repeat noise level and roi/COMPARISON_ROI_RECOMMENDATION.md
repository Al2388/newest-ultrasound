# Recommended C-scan Comparison ROI

Baseline used for mapping: `scan_2026-05-28_18-52-24`.

## Default: Balanced Core ROI

- Rows: `12:132` (120 rows)
- Columns: `35:465` (430 columns)
- Physical region: X `5.61-74.39 mm`, Y `6.04-65.96 mm`
- Area kept: `71.7%` of full C-scan
- Stable pixels within this ROI using all thresholds: `76.9%`
- Thresholds: amplitude span <= `0.55 V`, ToF span <= `0.18 us`, energy span <= `32`.

Noise span inside balanced ROI:

| feature | median span | p95 span | p99 span | max span |
|---|---:|---:|---:|---:|
| amplitude | 0.0800312 | 1.76416 | 3.75702 | 4.96283 |
| tof | 0.0379019 | 0.657182 | 2.54115 | 5.64819 |
| energy | 5.6689 | 97.0053 | 173.99 | 294.682 |

## Strict ROI

- Rows: `15:129`; columns: `50:450`
- Physical region: X `8.02-71.98 mm`, Y `7.55-64.45 mm`
- Area kept: `63.3%`; stable pixels within ROI: `78.4%`
- Use this stricter ROI for sensitive pixel-wise ToF or amplitude claims.

## Files

- `comparison_roi_overlay.png`
- `comparison_roi_noise_score.png`
- `comparison_roi_stable_mask.png`
- `comparison_roi_masks.npz`
- `comparison_roi_recommendation.json`
