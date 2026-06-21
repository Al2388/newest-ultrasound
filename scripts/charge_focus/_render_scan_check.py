"""Visual confirmation: render scan 68 and scan 87 (the two patched dropouts)
after the third-pass fix, in turbo-cmap style identical to the GIF frames."""
import numpy as np
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJ = Path("d:/SIOT/Ultrasound-Imaging-for-Li-ion-Battery-main")
sys.path.insert(0, str(PROJ / "scripts" / "longrun_analysis"))

d = np.load(PROJ / "reports/longrun_cycling_35c_charge_focus/_cache/stack.npz")
roi = d["roi_mask"]
amp = d["amplitude"]; tof = d["tof"]; eng = d["energy"]
x_mm = d["x_mm"]; y_mm = d["y_mm"]
extent = [x_mm.min(), x_mm.max(), y_mm.max(), y_mm.min()]

OUT = PROJ / "reports/longrun_cycling_35c_charge_focus/_patched_dropout_check.png"

fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
for col, (s, label) in enumerate([(68, "scan 68 (was Y=25.68mm dropout)"),
                                   (87, "scan 87 (was Y=40.78mm dropout)")]):
    # show only ROI-relevant columns
    cmap = plt.colormaps["turbo"].copy(); cmap.set_bad("white")
    for row, (mod, arr) in enumerate([("amp", amp), ("tof", tof), ("eng", eng)]):
        a = arr[s]
        v = a[roi]
        vmin = float(np.percentile(v, 5))
        vmax = float(np.percentile(v, 95))
        ax = axes[col, row]
        ax.imshow(a, extent=extent, aspect="equal", cmap=cmap,
                  vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
        ax.set_title(f"{label} -- {mod}", fontsize=10)
        ax.set_xlabel("X [mm]"); ax.set_ylabel("Y [mm]")

fig.suptitle("After third-pass dropout patch -- scan 68 & 87 should now look clean",
             fontsize=12)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
