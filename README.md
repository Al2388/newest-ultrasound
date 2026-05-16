# Ultrasound Imaging for Li-ion Battery

IoT acquisition system for ultrasound characterisation of Li-ion batteries.
A FastAPI web server orchestrates three acquisition modes against shared
hardware (a TiePie HS5 oscilloscope and a modified Ender 3D printer mounting
the transducer):

| Mode    | What it does                                                      | Output                      |
|---------|-------------------------------------------------------------------|-----------------------------|
| C-scan  | 2-D raster scan over the cell surface                             | Amplitude / ToF / Energy maps |
| A-scan  | Continuous single-point monitoring during cycling                 | HDF5 time-series + SOC annotations |
| Gauging | Live ToF readout                                                  | HDF5 stream                |

## Quick start

```bash
# from a fresh clone
python -m venv .venv
.venv\Scripts\activate              # Windows; on Linux/macOS use source .venv/bin/activate
pip install -e .                    # editable install of the ultrasound_battery package
ultrasound-battery                  # starts the FastAPI server on http://localhost:8000
```

The first launch prints a random `API_KEY`; copy it into `.env` so it
persists. See [`.env.example`](.env.example) for all configuration knobs.

For remote access:

```bash
tools/ngrok.exe http 8000
```

## Repository layout

```
.
├── src/ultrasound_battery/      # importable Python package
│   ├── app/                     # FastAPI: server.py + templates/
│   ├── services/                # scanner.py, ascan.py, gauging.py
│   ├── hardware/                # hs5.py, printer.py (instrument drivers)
│   └── cloud/                   # manager.py (Box / S3 archival)
├── scripts/                     # CLI runners for offline analysis
├── data/
│   ├── raw/                     # all raw measurements (acquisitions + external inputs); gitignored
│   │   ├── cscan/  ascan/  gauging/    # outputs of THIS instrument
│   │   └── cycler/  temperature/       # external (Autolab XLSX, PicoLog CSV)
│   └── processed/               # derived analysis outputs; gitignored
├── reports/
│   ├── experiments/             # per-experiment writeups (markdown + figures)
│   └── figures/                 # figures cited in reports
├── notebooks/                   # exploratory Jupyter work
├── references/                  # datasheets, calibration notes
├── tests/
├── docs/
├── tools/ngrok.exe              # operational binaries
├── pyproject.toml               # build + dependencies (PEP 621)
├── CITATION.cff                 # how to cite this code
└── .env.example
```

## Layout rationale

The layout follows current research-software conventions:

- **`src/` layout** — recommended by the [Python Packaging Authority](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) and PyOpenSci. Forces an explicit install before imports resolve, so the FastAPI server and standalone analysis scripts cannot accidentally rely on cwd being the repo root.
- **`data/raw/` vs `data/processed/`** — from [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/). Raw data (whether from our instrument or external loggers) is immutable; everything else is regenerable.
- **`scripts/` vs `src/`** — from [The Good Research Code Handbook](https://goodresearch.dev/setup). Importable, reusable code goes in the package; runnable entry-points go in `scripts/`.
- **`reports/experiments/<id>/`** — per-experiment writeups live with their figures, keeping the audit trail of "which plots came from which run" intact.

## License

TBD.

## Citation

If you use this code in academic work, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff).
