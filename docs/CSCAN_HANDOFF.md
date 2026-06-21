# C-scan handoff

This repository runs an ultrasound C-scan rig for an oil-immersed LFP pouch cell.
It is a lab-control project first and an analysis archive second: the important
handoff path is the FastAPI dashboard plus the C-scan acquisition service.

The final thesis/presentation framing is:

- System: pulse-echo immersion C-scan of a fixed LFP pouch cell.
- Hardware: CTS-8077PR pulser-receiver, 2.5 MHz focused immersion transducer,
  TiePie HS5 digitiser at 20 MHz, modified Ender-3 XY gantry, silicone-oil bath,
  Maccor Series 4000 cycler, PicoLog USB TC-08 temperature logger.
- Protocol: discharge baseline to 2.5 V, charge at C/10 to nominal 20, 40, 60,
  and 80 percent SOC plateaus, rest 2 h at each plateau, scan about every 9 min
  during rest, log OCV and cell temperature.
- Scientific caution: the maps are SOC-sensitive acoustic feature maps, not
  calibrated local SOC maps. Rest time and temperature are part of the
  measurement.

## Golden path

1. Put the probe and cell in the known physical start position.
2. Start the web app:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ultrasound-battery
   ```

3. Open `http://localhost:8000`.
4. Confirm the dashboard API key works. The key should come from `.env`.
5. Confirm live camera and TC-08 temperature are sane.
6. Run one short or normal C-scan from the dashboard.
7. Check the three live maps: amplitude, ToF, energy.
8. Confirm the output folder contains:

   ```text
   data/raw/cscan/cscan_<name>_<timestamp>/
     scan_amp.png
     scan_tof.png
     scan_eng.png
     scan_<timestamp>.npz
     scan_<timestamp>_meta.json
     session_manifest.json
     lines_raw/line_0000.npz ...
   ```

9. For repeated scans, use `scripts/run_cscan_noise_floor_batch.py` or
   `scripts/run_cscan_long_batch.py` after the web app is already running.

## Software setup

Use Python 3.10 or newer. On the lab Windows machine:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,dev]"
```

The acquisition package uses a `src/` layout, so install it in editable mode
before running scripts that import `ultrasound_battery`.

Create `.env` from `.env.example`:

```powershell
copy .env.example .env
```

At minimum set:

```text
API_KEY=<strong random string>
```

Optional:

```text
USBTC08_DLL=D:\SIOT\PicoLog\usbtc08.dll
BOX_DEV_TOKEN=
BOX_JWT_CONFIG=
BOX_FOLDER_ID=0
```

Note: `src/ultrasound_battery/app/server.py` now loads `.env` before reading
`API_KEY`, so a key in `.env` is persistent across server restarts.

## Hardware map

The main code paths are:

| Hardware | Code | Purpose |
|---|---|---|
| Ender-3 XY gantry | `src/ultrasound_battery/hardware/printer.py` | Opens Marlin serial port, sets G-code motion parameters, scans XY lines. |
| TiePie HS5 | `src/ultrasound_battery/hardware/hs5.py` | Streams CH1 echo and CH2 sync, extracts gated pulse features. |
| USB TC-08 | `src/ultrasound_battery/services/tc08.py` | Logs cell temperature and provides per-line temperature stats. |
| FastAPI server | `src/ultrasound_battery/app/server.py` | Dashboard/API entry point and mutual exclusion between acquisition modes. |
| C-scan service | `src/ultrasound_battery/services/scanner.py` | The actual raster scan worker. |

Wiring assumed by the code:

- HS5 CH1: receive echo from pulser-receiver, AC coupled.
- HS5 CH2: TTL sync/trigger pulse, DC coupled.
- Printer serial port: default `COM6`, 115200 baud.
- TC-08: channel list defaults to `[1]`; dashboard defaults to thermocouple type
  `T`, while the final thesis text says K-type. Confirm the actual probe type
  before running a report-quality experiment.

## Default C-scan settings

These are the settings used by the final workflow and dashboard:

| Setting | Value |
|---|---:|
| Scan area | 80 mm x 72 mm |
| Pitch | 0.5 mm |
| Scan speed | 25 mm/s |
| Acceleration mode | auto |
| Resolved acceleration at 25 mm/s | about 800 mm/s^2 |
| Output columns | 500 |
| Lines | 144 |
| Sample rate | 20 MHz |
| Gate | 25 to 50 us |
| CH1 range | 5 V for C-scan |
| Feature mode | Hilbert-envelope peak |
| Full scan time | about 9 min |

Do not casually change these for comparable data. The thesis conclusion depends
on repeat maps with the same physical origin, scan geometry, gain/range, gate,
temperature logging, and analysis ROI.

## Physical alignment

The scanner defines the current gantry position as `(0, 0)` at the start of a
fresh scan (`G92 X0 Y0`). That means the operator must physically place the
probe/cell in the known start position before pressing Start.

The software then uses a 2 mm X safety margin internally. The historical
analysis coordinates are the saved `x_mm` and `y_mm` arrays in each NPZ, not
raw Marlin coordinates. If the tank, cell, gantry, fixture, or origin changes,
run a fresh repeatability/noise-floor batch and re-check ROI placement.

Useful alignment check:

```powershell
python scripts\check_tank_alignment.py
```

## How the C-scan works

`CScanService._worker()` is the core acquisition loop:

1. Open printer and HS5.
2. Calibrate HS5 sync thresholds and measured PRF from CH2.
3. Preallocate NaN-filled amplitude, ToF, and energy arrays.
4. For each scan line:
   - move to line start;
   - command a Marlin `G1` line move;
   - stream HS5 data while the stage moves;
   - detect CH2 sync edges;
   - crop CH1 gate windows;
   - compute amplitude, ToF, and energy for each pulse;
   - use pulse timestamps plus a trapezoid motion model to map pulses to X;
   - reverse RTL lines back into left-to-right image coordinates;
   - align rows by phase correlation to reduce bidirectional striping;
   - save per-line raw data in `lines_raw/`;
   - update live PNGs every 5 lines.
5. Return to origin.
6. Save final `scan_*.npz`, `scan_*_meta.json`, and `session_manifest.json`.

The bidirectional raster is intentional. Positions in a single row are nearly
simultaneous; acquisition time mainly changes along Y. That is why the final
thesis makes tab-proximal/tab-distal comparisons along X.

## Output schema

Final `scan_*.npz` contains:

| Array | Meaning |
|---|---|
| `amplitude` | 2-D envelope-peak map, V |
| `tof` | 2-D time from gate start to envelope peak, us |
| `energy` | 2-D sum of squared gated echo samples |
| `x_mm`, `y_mm` | Analysis coordinate axes |
| `line_perf_start_s`, `line_perf_end_s` | Python `perf_counter()` line windows |
| `line_unix_start_s`, `line_unix_end_s`, `line_unix_center_s` | Real-time line windows |
| `line_pulse_count` | Pulse count per line |
| `line_temperature_mean_c`, `line_temperature_min_c`, `line_temperature_max_c`, `line_temperature_n` | TC-08 summary per line |

Each `lines_raw/line_XXXX.npz` stores pulse-level arrays:

- `waveforms`: gated CH1 windows if `save_waveforms=True`;
- `amplitude`, `tof_us`, `energy`;
- `timestamps` and `pulse_unix_s`;
- `x_mm`, `y_mm`, `direction`;
- per-line temperature and timing scalars.

The metadata JSON is the reproducibility record. Keep it beside the NPZ.

## Running batches

Noise floor / repeatability batch:

```powershell
python scripts\run_cscan_noise_floor_batch.py ^
  --base-url http://127.0.0.1:8000 ^
  --runs 6 ^
  --prefix baseline_repeat ^
  --voltage-v 3.232
```

Long rest/cycling batch:

```powershell
python scripts\run_cscan_long_batch.py ^
  --base-url http://127.0.0.1:8000 ^
  --duration-hours 14 ^
  --prefix longrun_cycling_35c ^
  --min-free-gb 20
```

Both scripts expect the FastAPI server to already be running and read the API
key from `.env`. The long-batch runner writes a checkpoint so it can resume.

## Final analysis conventions

Use this final 35 deg C thesis analysis folder for the submitted-result logic:

```text
reports/longrun_cycling_35c_charge_focus/thesis_spec_analysis/
```

The thesis-specific geometric ROI is:

```text
X = 14.6 to 64.5 mm
Y = 16.1 to 55.4 mm
```

Subregions:

```text
tab-distal   X = 14.6 to 30.0 mm
interior     X = 30.0 to 50.0 mm
tab-proximal X = 50.0 to 64.5 mm
```

Primary noise floors used for significance:

```text
ToF       sigma_ROI = 4.70 ns
Amplitude sigma_ROI = 2.22 mV
Energy    sigma_ROI = 0.081 a.u.
```

The earlier six-scan corrected-start ROI/noise folder is still important:

```text
reports/report c-scan baseline repeat noise level and roi/
```

It records the corrected-start repeatability work and an earlier strict
rectangular ROI recommendation. Use it when auditing the noise-floor/ROI
development, but use the thesis-specific 50 x 40 mm ROI above when reproducing
the final report figures.

## Temperature and cycler alignment

Never align ultrasound, cycler, and temperature streams by row number or
relative elapsed time alone. Align by real-world timestamps:

- C-scan: `line_unix_*` arrays in each NPZ.
- TC-08: `unix_s` and `timestamp_iso` in `temperature_log.csv`.
- Cycler: exported time/voltage/current/SOC-like columns.

The final thesis used the measured cell temperature, not bath or heater
temperature. The effective ToF temperature coefficient near the operating point
was about:

```text
alpha_T = +73.5 to +73.6 ns/deg C
```

Treat this as a local system sensitivity, not a universal material constant.

## Quality checks before trusting a run

Check these immediately after a scan:

- All 144 lines completed unless the scan was intentionally stopped.
- `line_pulse_count` is nonzero and roughly consistent across lines.
- `line_temperature_n` is nonzero for most/all lines when TC-08 is connected.
- Live amplitude map resolves the cell and fixture in the expected place.
- ToF map has no large blank rows, diagonal jumps, or obvious row swaps.
- `scan_*_meta.json` reports the expected gate, PRF, speed, pitch, acceleration,
  and `partial=false`.
- Free disk space is enough if `save_waveforms=True`.

If maps look shifted relative to old runs, do not "fix it in analysis" first.
Physically re-check origin, fixture, transducer standoff, and tank position.

## Known gotchas

- `.env` API keys only worked after adding `load_dotenv()` to `server.py`.
- Cloud upload is optional and fail-safe. Current `CloudManager` imports
  `box_sdk_gen`; if that package is absent, scans still run offline.
- `pyproject.toml` lists `box-sdk-gen` for Box archival, but the feature remains
  optional unless Box credentials are configured.
- The dashboard thermocouple default is `T`; the final thesis text says K-type.
  Confirm the real probe type before collecting final data.
- The C-scan Y travel behavior is historically calibrated around the current
  80 x 72 mm geometry. Revalidate travel limits before increasing scan height.
- The row-alignment code is tuned for the existing gantry and scan speed. If
  speed, acceleration, belt tension, or fixture mass changes, rerun repeatability.
- Do not compare amplitude across sessions if pulser gain, receiver gain, CH1
  range, coupling, or standoff changed.

## Online references used for this handoff

- FastAPI official docs: running an ASGI app with Uvicorn and binding to
  `0.0.0.0`: https://fastapi.tiangolo.com/deployment/manually/
- Python Packaging User Guide: `src/` layout requires installation and avoids
  accidental imports from the repo root:
  https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/
- Montoya-Bedoya et al. review, 2026: ultrasonic testing is promising for
  non-invasive lithium-ion battery assessment, but interpretation still needs
  careful links between acoustic signals and electrochemical/elastic behavior:
  https://arxiv.org/abs/2601.08075
- Montoya-Bedoya et al., 2021: ultrasound TOF/SOS/amplitude can respond to
  battery state, but response can be complex and not always monotonic:
  https://arxiv.org/abs/2110.14033

Vendor pages to check before rebuilding the machine:

- TiePie HS5 / Handyscope HS5 support and drivers: https://www.tiepie.com/
- Pico Technology USB TC-08 drivers and SDK: https://www.picotech.com/
