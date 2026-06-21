# Report-Ready C-scan Noise/ROI Assets

Use these figures and tables for the report section that defines the C-scan
repeatability noise floor and the preliminary ROI for later model fitting.

## Recommended Figures

- `fig01_temperature_stability.png`: confirms temperature stability during the 10 repeated scans.
- `fig02_roi_definition_on_baseline.png`: shows the selected ROI and stable pixels on the baseline C-scan.
- `fig03_noise_floor_span_maps.png`: shows pixel-wise max-min noise-floor span for amplitude, ToF, and energy.
- `fig04_repeatability_sigma_maps.png`: shows pixel-wise standard deviation across the 10 scans.
- `fig05_pixel_noise_sigma_distributions.png`: compares local noise distributions in the ROI and stable mask.
- `fig06_roi_mean_repeatability.png`: shows scan-to-scan drift of ROI-averaged features.
- `fig07_battery_focused_roi.png`: restricts the analysis region to the battery body and its stable pixels.

## Recommended Tables

- `table01_scan_conditions.*`: scan IDs, baseline selection, and temperature conditions.
- `table02_roi_definition.*`: ROI coordinates and physical dimensions.
- `table03_noise_floor_by_roi.*`: detailed noise-floor statistics for each ROI.
- `table04_model_noise_terms.*`: compact values to use in preliminary model fitting.
- `table05_battery_focused_noise_terms.*`: compact values for the battery-focused ROI.

Suggested report logic:

1. Show temperature stability first.
2. Define the balanced ROI and stable mask.
3. Quantify noise using sigma and span.
4. Use the stable-mask median sigma as representative pixel-level noise.
5. Use the balanced ROI scan-mean sigma as the ROI-averaged repeatability limit.
6. For battery-only comparisons, use the battery body ROI and battery stable mask.
