import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
meta = pd.read_csv(PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/meta.csv")

cases = [(60, 82, "scan #60 row 82"), (80, 76, "scan #80 row 76")]

for scan_idx, row, label in cases:
    m = meta.iloc[scan_idx]
    sess = PROJ / m["session_dir"].replace("\\", "/")
    print(f"\n=== {label} ===")
    print(f"session: {sess.name}")
    line = np.load(sess / "lines_raw" / f"line_{row:04d}.npz")
    x_pulse = line["x_mm"]
    amp_pulse = line["amplitude"]
    print(f"  pulses total: {len(x_pulse)}")
    print(f"  x_mm NaN count: {int(np.isnan(x_pulse).sum())}")
    print(f"  amp NaN count: {int(np.isnan(amp_pulse).sum())}")
    finite_x = x_pulse[np.isfinite(x_pulse)]
    if len(finite_x):
        print(f"  x_mm valid range: {finite_x.min():.2f} -> {finite_x.max():.2f} mm")
    print(f"  direction: {int(line['direction'])}")
    isnan_x = np.isnan(x_pulse)
    if isnan_x.any():
        first_nan = int(np.argmax(isnan_x))
        prev_x = float(x_pulse[first_nan - 1]) if first_nan > 0 else float("nan")
        print(f"  first NaN x at pulse #{first_nan} (prev x = {prev_x:.2f} mm)")
        after = isnan_x[first_nan:]
        print(f"  pulses #{first_nan}..end: total {len(after)}, NaN x {int(after.sum())}, "
              f"valid x {int((~after).sum())}")
        print(f"  -> ALL trailing pulses have NaN x: {bool(np.all(after))}")
