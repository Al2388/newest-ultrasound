# `data/` — measurement and derived artefacts

This directory is **gitignored**. Real data is large (GB+) and is archived
separately (Box / S3 via [`cloud.manager`](../src/ultrasound_battery/cloud/manager.py)).
Only this README and the directory skeleton are tracked.

## Layout

```
data/
├── raw/                # immutable inputs — never edit in place
│   ├── cscan/          # this project's C-scan acquisitions (NPZ + meta JSON + PNGs + lines_raw/)
│   ├── ascan/          # this project's A-scan sessions       (HDF5 + annotations JSON)
│   ├── gauging/        # this project's gauging sessions       (HDF5)
│   ├── cycler/         # external: Autolab cycler XLSX exports
│   └── temperature/    # external: PicoLog temperature CSV exports
└── processed/          # derived analysis outputs (regenerable from raw/)
```

The split between `raw/` and `processed/` follows
[Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
and [Wilson et al., "Best Practices for Scientific Computing"](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745):
treat raw data as **immutable**, and treat everything in `processed/` as
**regenerable** by re-running the scripts in [`scripts/`](../scripts/).

## Provenance

| Folder                  | Source                               | Schema authority                          |
|-------------------------|--------------------------------------|-------------------------------------------|
| `raw/cscan/`            | This system (`services.scanner`)     | `scan_*.npz` + `scan_*_meta.json`         |
| `raw/ascan/`            | This system (`services.ascan`)       | HDF5, see `services/ascan.py` docstring   |
| `raw/gauging/`          | This system (`services.gauging`)     | HDF5, see `services/gauging.py` docstring |
| `raw/cycler/`           | Autolab potentiostat (external)      | XLSX with columns: Time, WE(1).Potential, WE(1).Current, ... |
| `raw/temperature/`      | PicoLog logger (external)            | CSV with `HH:MM:SS` timestamp + channel temperatures |
