"""Quick capacity readout from Maccor tab-delimited cycle text files."""
import sys
from pathlib import Path

paths = [
    Path(r"d:\SIOT\Ultrasound-Imaging-for-Li-ion-Battery-main\data\raw\cycler\19-5 cycle.txt"),
    Path(r"d:\SIOT\Ultrasound-Imaging-for-Li-ion-Battery-main\data\raw\cycler\21-5 cycle.txt"),
]
if len(sys.argv) > 1:
    paths = [Path(p) for p in sys.argv[1:]]


def summarise(path: Path) -> None:
    print(f"=== {path.name} ===")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # header info (first ~5 lines)
    for line in lines[:5]:
        line = line.rstrip("\n").strip()
        if line:
            print(" ", line)

    # find header
    hdr_idx = None
    for i, line in enumerate(lines[:20]):
        if line.startswith("Rec\t"):
            hdr_idx = i
            break
    if hdr_idx is None:
        print("  (no header row found)")
        return
    cols = [c.strip() for c in lines[hdr_idx].rstrip("\n").split("\t")]
    idx = {c: i for i, c in enumerate(cols)}

    modes = {"C": "CHARGE", "D": "DISCHARGE"}
    stats = {m: {"n": 0, "cap_max": 0.0, "i_sum": 0.0,
                 "i_max": 0.0, "i_min": 1e9,
                 "v_max": 0.0, "v_min": 1e9,
                 "t_first": None, "t_last": None} for m in modes}
    rest_n = 0
    total_rows = 0

    for line in lines[hdr_idx + 1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(cols):
            continue
        total_rows += 1
        md = parts[idx["MD"]].strip()
        if md == "R":
            rest_n += 1
            continue
        if md not in stats:
            continue
        try:
            cap = float(parts[idx["Capacity"]])
            curr = float(parts[idx["Current"]])
            volt = float(parts[idx["Voltage"]])
        except ValueError:
            continue
        tt = parts[idx["Test Time"]].strip()
        s = stats[md]
        s["n"] += 1
        s["cap_max"] = max(s["cap_max"], cap)
        s["i_sum"] += abs(curr)
        s["i_max"] = max(s["i_max"], abs(curr))
        s["i_min"] = min(s["i_min"], abs(curr))
        s["v_max"] = max(s["v_max"], volt)
        s["v_min"] = min(s["v_min"], volt)
        if s["t_first"] is None:
            s["t_first"] = tt
        s["t_last"] = tt

    print(f"  rows: {total_rows} (rest: {rest_n})")
    for md, label in modes.items():
        s = stats[md]
        if s["n"] == 0:
            print(f"  {label}: no rows")
            continue
        i_avg = s["i_sum"] / s["n"]
        print(
            f"  {label}: n={s['n']}, cap_max={s['cap_max']*1000:.1f} mAh, "
            f"I=[{s['i_min']*1000:.1f}-{s['i_max']*1000:.1f}] mA (avg {i_avg*1000:.1f}), "
            f"V=[{s['v_min']:.3f}-{s['v_max']:.3f}] V, "
            f"time {s['t_first']} -> {s['t_last']}"
        )
    print()


for p in paths:
    summarise(p)
