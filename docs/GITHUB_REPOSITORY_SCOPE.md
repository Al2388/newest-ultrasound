# GitHub Repository Scope

This repository should be a runnable handoff for C-scan acquisition plus a
small, readable evidence package. It should not try to be the full experiment
archive.

## Essential Code

- `src/ultrasound_battery/app/server.py` and `templates/index.html`: FastAPI
  dashboard, API-key auth, C-scan/A-scan/gauging routes, TC-08 routes, and live
  camera stream.
- `src/ultrasound_battery/services/scanner.py`: C-scan orchestration, raster
  paths, waveform capture, map generation, HDF5 output, PNG summaries, and Box
  upload hooks.
- `src/ultrasound_battery/hardware/hs5.py`: TiePie HS5 stream setup, Hilbert
  envelope, and peak feature extraction.
- `src/ultrasound_battery/hardware/printer.py`: serial stage motion, homing,
  absolute moves, and guarded G-code communication.
- `src/ultrasound_battery/services/tc08.py`: Pico TC-08 temperature logging and
  CSV output.
- `src/ultrasound_battery/services/camera.py`: optional webcam/MJPEG rig view.
- `src/ultrasound_battery/services/ascan.py` and `gauging.py`: single-point
  diagnostics and probe-height/gauge workflows.
- `src/ultrasound_battery/cloud/manager.py`: optional Box archival; scans still
  run if Box credentials or SDK are absent.
- `scripts/run_cscan_noise_floor_batch.py` and `run_cscan_long_batch.py`:
  repeat/long-run operators for unattended acquisition.
- `scripts/check_tank_alignment.py`, `render_cscan_time_mapping.py`,
  `match_cscan_context.py`, and `match_cscan_picolog_temperature.py`: core
  post-acquisition helpers.
- `scripts/charge_focus/spec_thesis_rest_analysis.py` and
  `spec_thesis_figures_v2.py`: thesis-grade 35 C analysis and figure export.

## Results/Data To Show

Show only the compact, interpretable outputs needed to prove the method:

- `reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/`: final
  35 C charge-focus result tables, summary JSON, and report-ready figures.
- `reports/longrun_cycling_35c_charge_focus/README.md` and `ROBUSTNESS.md`:
  narrative and robustness checks for the 35 C result.
- `reports/longrun_cycling_35c_charge_focus/_cache/stack.npz` and `meta.csv`:
  compact analysis-ready 35 C maps and per-scan metadata already committed on
  this branch. These are acceptable as a small reproducibility bundle.
- `reports/report c-scan baseline repeat noise level and roi/`: selected
  repeatability/noise-floor report assets and tables, especially
  `USE_THIS_FOR_FUTURE_ANALYSIS.md`, `NOISE_FLOOR_REPORT.md`, and the
  `report_ready_figures_tables/` figures/tables.

## Keep Out Of Git

- Raw C-scan waveform folders under `data/raw/cscan/`.
- Analysis caches such as `*.npz` stacks unless deliberately released as a
  separate dataset artifact.
- GIF animations, full QA galleries, and exploratory one-off report folders.
- `.env`, Box credentials, PicoLog exports with private paths, and local scratch
  folders such as `.codex_tmp/`.

If a future reader needs to reproduce every plot from raw waveforms, publish the
raw data separately through Box, Zenodo, OSF, or institutional storage, then add
a DOI/link in `data/README.md`.
