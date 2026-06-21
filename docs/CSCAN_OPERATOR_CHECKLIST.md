# C-scan operator checklist

Use this at the bench. For the deeper code/data explanation, read
`docs/CSCAN_HANDOFF.md`.

## Before power-on

- Cell is fixed in the same fixture orientation as the reference runs.
- Transducer standoff/focus has not been changed, or has been recalibrated.
- Oil level covers the acoustic path and there are no visible bubbles on the
  cell/probe face.
- Pulser-receiver settings and gain match the reference experiment.
- HS5 CH1 is echo signal, HS5 CH2 is sync pulse.
- Ender USB is connected and expected to be `COM6`.
- TC-08 thermocouple is attached to the cell and the thermocouple type is
  correct in the dashboard.
- Cycler logging is started if the scan will be matched to SOC/OCV.
- Disk has enough free space for `lines_raw/` waveform archives.

## Start the app

```powershell
cd D:\SIOT\Ultrasound-Imaging-for-Li-ion-Battery-main
.\.venv\Scripts\Activate.ps1
ultrasound-battery
```

Open:

```text
http://localhost:8000
```

If the server prints a generated API key, put a persistent key into `.env`
before collecting real data.

## Standard scan settings

```text
ROI width       80.0 mm
ROI height      72.0 mm
Resolution       0.5 mm
Speed           25.0 mm/s
Acceleration     Auto from speed
Manual accel    800 mm/s^2 if manual is required
Output columns  500
Gate             25 to 50 us
```

Use a clear scan name:

```text
<cell>_<condition>_<soc/rest/run>
```

Example:

```text
lifun35c_soc40_rest_r03
```

## Run one scan

1. Put the probe at the known physical origin.
2. On C-Scan tab, confirm settings.
3. Check the persistent camera/temperature overlay.
4. Press `Start Scan`.
5. Watch first few rows of amplitude, ToF, and energy.
6. If the probe path is wrong, press `Emergency Stop`; wait for the current
   line to finish and return to origin.
7. Let the scan complete: expected `144 / 144`, about 9 min.

## After each scan

Check the newest folder under:

```text
data/raw/cscan/
```

Required files:

```text
scan_amp.png
scan_tof.png
scan_eng.png
scan_<timestamp>.npz
scan_<timestamp>_meta.json
session_manifest.json
lines_raw/
```

Quick sanity:

- `session_manifest.json` state is `completed`.
- Metadata says `partial=false`.
- All three PNGs have a plausible cell outline.
- No obvious blank stripe or shifted half of the map.
- TC-08 status shows recent temperature samples.

## Repeated scans

Start the web app first, then run:

```powershell
python scripts\run_cscan_noise_floor_batch.py --runs 6 --prefix baseline_repeat
```

For long rest/cycling experiments:

```powershell
python scripts\run_cscan_long_batch.py --duration-hours 14 --prefix longrun_cycling_35c --min-free-gb 20
```

Leave the lab note with:

- scan prefix and exact start time;
- cell voltage/SOC condition;
- bath/cell temperature setpoint and observed range;
- any physical intervention;
- any stopped/partial scan folder names.

## Stop / recover

- Normal end: the probe returns to origin and status becomes `COMPLETED`.
- Early stop: press `Emergency Stop`, wait for return, then inspect partial
  output before deciding whether to delete or label it as partial.
- If the printer is disconnected or unresponsive, stop the server, power-cycle
  the printer if needed, then restart the server.
- If HS5 is not found, close other TiePie/Pico tools and reconnect USB.
- If TC-08 is missing, verify Pico drivers and `USBTC08_DLL`, then restart
  temperature logging.

## Do not change during a comparable run

- Physical origin.
- Cell fixture position.
- Transducer standoff.
- Pulser/receiver gain.
- HS5 channel range.
- Gate window.
- Scan geometry.
- Temperature setpoint/control method.
- Analysis ROI.

Changing any of these means the next scans need a fresh repeatability/noise
floor check before they should be compared to the thesis data.
