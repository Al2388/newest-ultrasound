# 13-5 Combined C/10 Validation Checklist

## Files
- `Battery Cycler Data/13-5.xlsx`
- `Battery Cycler Data/13-5 (2).xlsx`
- `pico temperature/13-5.csv`

## Dataset Coverage
- Combined cycler rows: 14,328
- Combined raw timestamp span: 20.899 h
- Nominal capacity assumed: 0.860 Ah
- Gaps >30 s between cycler rows: 1
  - gap after row 7229: 1.002 h, V endpoint 3.664->3.383 V

## Detected Segments
- charge: rows 0-7229, 10.040 h, I_mean=86.242 mA, V=2.602->3.664 V, source `Battery Cycler Data/13-5.xlsx`
- discharge: rows 7230-14327, 9.857 h, I_mean=-85.639 mA, V=3.383->2.473 V, source `Battery Cycler Data/13-5 (2).xlsx`

## Critical Checks
### 1. Voltage profile shape: PASS with caveat
Charge is smooth rising and discharge is smooth falling. Large wrong-direction steps >10 mV: charge=0, discharge=0. There is an unlogged gap/rest of 1.002 h between exports.

### 2. Capacity check vs rated: PASS
Q_charge=0.8659 Ah (100.7% nominal); Q_discharge=0.8441 Ah (98.2% nominal).

### 3. Coulombic efficiency: WARN
CE=97.49% using Q_discharge/Q_charge. By your thresholds this is WARN.

### 4. Voltage limits respected: PASS for broad LFP 2.0-3.65; WARN vs 2.5V discharge cutoff
Observed V=2.473-3.664. Charge exceeds 3.65 V by 13.6 mV at cutoff; discharge reaches 27.5 mV below 2.5 V cutoff threshold, but remains above 2.0 V.

### 5. Current control quality: PASS
Charge median=86.243 mA, max dev=0.142% (0.122 mA). Discharge median=-85.632 mA, max dev=0.178% (0.153 mA). No explicit zero-current rest rows; only an unlogged gap between files.

### 6. Temperature behaviour: PASS
Across combined cycler timestamps: 21.279-22.471 C, range=1.192 C, peak-start=0.176 C. Full temp file range=1.196 C over 23.06 h.

## Diagnostic Plots
- `01_voltage_current_vs_time_combined.png`
- `02_voltage_vs_capacity_combined.png`
- `03_dqdv_vs_voltage_combined.png`
- `04_temperature_voltage_vs_time_combined.png`
- `05_rest_gap_voltage_endpoints.png`

## Overall Decision
The combined charge-plus-discharge data is usable for the main C/10 validation, with two caveats: the rest/hold period is not logged continuously in the cycler exports, and CE is only 97.47%, which is below the >99% fresh-cell target and lands in the checklist warning band. Current control, temperature stability, voltage shape, and discharge capacity pass.
