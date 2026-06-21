# Project map

This is the shortest route through the codebase for a new maintainer.

## Runtime entry point

```text
pyproject.toml
  [project.scripts]
  ultrasound-battery = "ultrasound_battery.app.server:main"
```

`ultrasound-battery` starts `src/ultrasound_battery/app/server.py`, which
creates one global service instance for each acquisition mode:

- `temp = TC08Service()`
- `scanner = CScanService(temperature_service=temp)`
- `ascan = AScanService(cloud=scanner.cloud)`
- `gauge = GaugingService(cloud=scanner.cloud)`
- `camera = CameraService(...)`

The server enforces mutual exclusion between C-scan, A-scan, and gauging.
Temperature and camera can run alongside a scan.

## Acquisition modules

| File | Read this for |
|---|---|
| `src/ultrasound_battery/app/server.py` | API routes, request models, authentication, dashboard mount. |
| `src/ultrasound_battery/app/templates/index.html` | Single-page dashboard controls and polling code. |
| `src/ultrasound_battery/services/scanner.py` | C-scan orchestration, row gridding, row alignment, final NPZ/meta schema. |
| `src/ultrasound_battery/hardware/hs5.py` | TiePie stream-mode setup, sync calibration, pulse feature extraction. |
| `src/ultrasound_battery/hardware/printer.py` | Marlin serial control and XY motion commands. |
| `src/ultrasound_battery/services/tc08.py` | USB TC-08 ctypes wrapper and threaded CSV logger. |
| `src/ultrasound_battery/services/ascan.py` | Fixed-point A-scan monitoring and HDF5 output. |
| `src/ultrasound_battery/services/gauging.py` | Live ToF readout and HDF5 output. |
| `src/ultrasound_battery/cloud/manager.py` | Optional Box archival. Scans still run if this is disabled. |

## C-scan API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard. |
| `/api/start` | POST | Start C-scan. Requires `X-API-Key` and free hardware. |
| `/api/stop` | POST | Stop after current line and save partial data. |
| `/api/return` | POST | Return probe to origin when not scanning. |
| `/api/status` | GET | Live scan status and image URLs. |
| `/api/data/latest` | GET | Download latest C-scan NPZ. Requires key. |
| `/api/data/meta` | GET | Latest metadata JSON. |
| `/api/temp/start` | POST | Start TC-08 logger. |
| `/api/temp/status` | GET | TC-08 status and latest temperature. |

Minimum API example:

```powershell
$key = (Select-String -Path .env -Pattern '^API_KEY=').Line.Split('=',2)[1]
$body = @{
  scan_name = "smoke_test"
  roi_w = 80.0
  roi_h = 72.0
  pitch = 0.5
  speed = 25.0
  accel_mode = "auto"
  accel_mm_s2 = 800
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/start" `
  -Headers @{ "X-API-Key" = $key } `
  -Body $body `
  -ContentType "application/json"
```

## Batch and analysis scripts

Primary operator scripts:

| Script | Purpose |
|---|---|
| `scripts/run_cscan_noise_floor_batch.py` | Repeated C-scans for repeatability/noise-floor analysis. |
| `scripts/run_cscan_long_batch.py` | Long repeated C-scan series with checkpoint/resume and free-space guard. |
| `scripts/check_tank_alignment.py` | Compare a new scan to a reference for lateral shift and working-distance drift. |
| `scripts/render_cscan_time_mapping.py` | Re-render a scan from per-line pulse timestamps for timing diagnostics. |
| `scripts/match_cscan_context.py` | Match C-scan NPZ to cycler and temperature logs by timestamps. |

Thesis-specific analysis:

| Script | Purpose |
|---|---|
| `scripts/charge_focus/spec_thesis_rest_analysis.py` | Rebuild final thesis tables/summary/figures from cached 35 deg C data. |
| `scripts/charge_focus/spec_thesis_figures_v2.py` | Re-render thesis figures with cleaner style. |
| `scripts/charge_focus/analyze_35c_charge_focus.py` | Main 35 deg C charge-focus analysis package. |
| `scripts/charge_focus/analyze_35c_robustness.py` | Reference-region and sub-ROI robustness checks. |
| `scripts/charge_focus/run.py` | Thin convenience runner for the charge-focus stack. |

Many other scripts are one-off figure builders or diagnostics. Keep them, but
do not treat every script as an operator entry point.

## Important report/data folders

| Path | Meaning |
|---|---|
| `data/raw/cscan/` | Real C-scan acquisitions. Gitignored. |
| `data/raw/temperature/` | TC-08 logs. Gitignored. |
| `data/raw/cycler/` | External cycler exports. Gitignored. |
| `reports/report c-scan baseline repeat noise level and roi/` | Corrected-start six-scan noise/ROI basis. |
| `reports/longrun_cycling_35c_charge_focus/` | 35 deg C charge/rest analysis outputs. |
| `reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/` | Final thesis-spec summaries, CSVs, JSON, and figures. |

## Current working-tree note

This repository currently has many untracked data/report/script outputs and
modified acquisition files. Do not clean or reset the tree unless the project
owner explicitly asks. The large untracked report folders are part of the
project audit trail even when they are not committed.

## Maintainer checklist after code changes

- Run `python -m pytest` if tests have been added. At handoff time the `tests/`
  directory only contains `.gitkeep`.
- Run at least `python -m compileall src` after Python source edits.
- For scan-path changes, run a hardware smoke scan and inspect `scan_*_meta.json`.
- For geometry, gate, feature, or alignment changes, rerun a repeatability batch.
- For analysis changes, regenerate the relevant report folder and preserve the
  command in a README or manifest.

## Known technical debt

- No real automated tests yet.
- Box archival is optional and fail-safe; install/configure `box-sdk-gen` only
  if the new operator needs cloud upload.
- The final thesis scripts use absolute `D:/SIOT/...` paths; they run on the
  lab machine but are not portable without edits.
- The C-scan service is highly stateful and hardware-coupled, so most bugs only
  show up with the rig attached. Keep code changes small and validate at the
  bench.
- Some old Markdown/files show encoding mojibake for symbols such as arrows and
  microseconds. New docs should stay ASCII unless the file is cleaned globally.
