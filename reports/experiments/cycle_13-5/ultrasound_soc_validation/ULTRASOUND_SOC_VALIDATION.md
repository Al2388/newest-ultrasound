# 13-5 Ultrasound SoC Validation

## Purpose
Show whether the ultrasonic A-scan features track internal cell state-of-charge during the C/10 cycle, and whether the observed ToF changes are larger than can be explained by temperature drift alone.

## Inputs
- A-scan H5: `data/ascan/ascan_session_2026-05-12_15-42-53/ascan_session_2026-05-12_15-42-53.h5`
- Cycler: `Battery Cycler Data/13-5.xlsx`
- Cycler: `Battery Cycler Data/13-5 (2).xlsx`
- Temperature: `pico temperature/13-5.csv`
- Nominal capacity used for SoC: 0.860 Ah

## SoC Axis Note
- Initial SoC offset used in this run: 0.00%
- If the cell did not start at 0% SoC, the absolute SoC values are shifted by the unknown starting SoC. The `relative_soc_change_pct` column gives the measured coulomb-counted change from the first cycler row. The ultrasound conclusion depends on feature changes versus cycling state, so a constant initial-SoC offset changes the x-axis labels but not the ToF/temperature separation result.
- Because the charge step alone delivered 0.8659 Ah against a nominal 0.860 Ah, this export behaves like a near-empty-to-full charge on the 0.86 Ah capacity scale. If the cell truly had substantial starting SoC, then either the effective capacity/cutoff window is larger than 0.86 Ah, the starting SoC estimate is wrong, or the current/capacity calibration needs checking.

## Cycle Summary
- Charge capacity: 0.8659 Ah
- Discharge capacity: 0.8441 Ah
- Coulombic efficiency: 97.49%
- Combined cycler time span: 20.90 h
- Temperature span over cycler timestamps: 21.279 to 22.471 C
- ToF span over cycler timestamps: -524.0 to 0.0 ns
- P2P clipped-sample mean/max: 1.01% / 4.00%

## Temperature And SoC Separation Test
- Model used: ordinary least-squares regression on ToF. The SoC/state term uses coulomb-counted SoC with polynomial terms plus charge/discharge branch terms; the temperature term is linear in measured temperature.
- Temperature-only model R2 for ToF: 0.564
- SoC/state-only model R2 for ToF: 0.984
- SoC/state + temperature model R2 for ToF: 0.985
- R2 gained by adding SoC/state after temperature: 0.421
- R2 gained by adding temperature after SoC/state: 0.001
- Residual error removed by adding SoC/state after temperature: 96.5%
- Residual error removed by adding temperature after SoC/state: 5.1%
- Fitted temperature coefficient inside the combined model: -51.01 ns/C
- Model-estimated temperature contribution span: 60.8 ns
- Model-estimated SoC/state contribution span: 491.5 ns
- SoC/state contribution is about 8.1x larger than the temperature contribution in this run.
- Observed ToF span: 524.0 ns

Interpretation: the literature says temperature affects ultrasonic ToF, so it should be included and reported. In this dataset, however, the independent temperature term is small compared with the SoC/state term: SoC/state explains more variance than temperature alone, adds much more explanatory power when added after temperature, and has a larger fitted ToF span. The charge and discharge branches also separate at overlapping temperatures, which supports the conclusion that the acoustic response is tracking internal electrochemical/mechanical state, not only ambient drift.

## Robustness Check
- Repeating the same model comparison using A-scan snapshots as the table rows gives the same conclusion, so the result is not an artifact of weighting by cycler timestamps.
- A-scan-row temperature-only R2: 0.535
- A-scan-row SoC/state-only R2: 0.985
- A-scan-row SoC/state + temperature R2: 0.986
- A-scan-row SoC/state component span: 496.1 ns
- A-scan-row temperature component span: 68.3 ns

## Literature Alignment
- Hsieh et al. established the common electrochemical-acoustic ToF framing: acoustic ToF changes because lithiation changes density, modulus, attenuation, and mechanical state inside the cell. DOI: https://doi.org/10.1039/C5EE00111K
- Ke et al. used ToF and amplitude during lithium-ion pouch-cell cycling and explicitly reported that ToF is influenced by temperature, while amplitude can correlate with physical electrode-layer changes. DOI: https://doi.org/10.1016/j.jpowsour.2022.232031
- Borujerdi, Jin, and Zhu used voltage/current/SOC plus ultrasonic P2P amplitude and ToF shift during cycling, then applied in-situ temperature correction; after correction, ToF shift correlated well with SOC. DOI: https://doi.org/10.1016/j.jpowsour.2024.234103
- Zhang et al. treated SOC and temperature as jointly estimated states from ultrasonic reflection-wave features, which supports presenting temperature and SOC together rather than pretending temperature is irrelevant. DOI: https://doi.org/10.3390/batteries9060335
Compared with those papers, this analysis follows the typical structure: validate voltage/current cycling, compute coulomb-counted SOC, extract ToF shift plus amplitude/energy features, plot ToF versus time/SOC/voltage, include temperature correction, and state charge/discharge hysteresis. The main limitation is that this is one cycling experiment with only about 1.2 C of temperature movement; a stronger causal temperature calibration would need repeated cycles or a controlled temperature sweep at fixed SOC.

## How To Present This
1. Start with `01_overview_voltage_current_temp_tof.png`: voltage/current confirm the cycle, temperature shows only slow ambient drift, and ToF changes strongly through the electrochemical steps.
2. Use `02_tof_vs_soc_charge_discharge.png`: the ToF feature has a structured SoC dependence and a charge/discharge branch difference, which is expected for hysteretic cell mechanics.
3. Use `03_temperature_vs_soc_separation.png`: temperature alone gives a weaker explanation than SoC/state plus temperature.
4. Use `04_tof_temperature_soc_decomposition.png`: this is the clearest slide for your supervisor. It separates the fitted temperature and SoC/state ToF components and shows the SoC/state contribution is larger.
5. Use `05_temperature_corrected_tof_vs_soc.png`: after subtracting the fitted temperature term, the SoC trajectory remains.
6. Use `07_representative_waveforms_by_soc.png`: the actual echo packet shifts with cycling, so the feature is visible in the raw ultrasonic signal, not only in a derived number.

## Important Caveats
- The cycler rest between charge and discharge is present as an unlogged gap between two exports, so continuous rest-relaxation validation is not available from the cycler data.
- Peak-to-peak amplitude is affected by receiver clipping near the voltage rails. Energy and ToF are more reliable features for this run.
- The temperature coefficient is an in-run compensation estimate, not a chamber-calibrated material law. Use it to show temperature was checked and corrected, not as a universal coefficient for all cells.
- Coulombic efficiency is below the >99% fresh-cell target, so this cycle is useful as a demonstration run but should be repeated for calibration-grade data.

## Generated Figures
- `01_overview_voltage_current_temp_tof.png`
- `02_tof_vs_soc_charge_discharge.png`
- `03_temperature_vs_soc_separation.png`
- `04_tof_temperature_soc_decomposition.png`
- `05_temperature_corrected_tof_vs_soc.png`
- `06_amplitude_energy_clipping.png`
- `07_representative_waveforms_by_soc.png`
- `08_tof_voltage_energy_soc.png`

Aligned feature table: `aligned_ultrasound_features.csv`
