"""C-scan time-series animation across the 22h run.

Produces an MP4 per modality with overlays for run idx, time, SOC,
voltage, temperature, and step tag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT_ROOT, load, roi_bounds_mm  # noqa: E402

OUT = OUT_ROOT / "06_animation"
OUT.mkdir(parents=True, exist_ok=True)

UNITS = {"tof": "µs", "amplitude": "V", "energy": "(a.u.)"}
COLOR_BY_TAG = {"charge": "tab:red", "discharge": "tab:blue", "rest": "#444", "transition": "tab:orange"}


def _build_anim(stack, mod: str):
    arr = getattr(stack, mod)
    meta = stack.meta
    n = arr.shape[0]
    t_h = (pd.to_datetime(meta.time_utc) - pd.to_datetime(meta.time_utc.iloc[0])).dt.total_seconds().values / 3600

    vmin = float(np.nanpercentile(arr[:, stack.roi_mask], 2))
    vmax = float(np.nanpercentile(arr[:, stack.roi_mask], 98))

    x_lo, x_hi, y_lo, y_hi = roi_bounds_mm(stack)
    extent = [stack.x_mm.min(), stack.x_mm.max(), stack.y_mm.max(), stack.y_mm.min()]

    fig = plt.figure(figsize=(13, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1.0])
    ax_img = fig.add_subplot(gs[:, 0])
    ax_soc = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])
    ax_t = fig.add_subplot(gs[2, 1])

    im = ax_img.imshow(arr[0], extent=extent, aspect="equal", cmap="turbo", vmin=vmin, vmax=vmax)
    ax_img.plot([x_lo, x_hi, x_hi, x_lo, x_lo], [y_lo, y_lo, y_hi, y_hi, y_lo], "k--", lw=1, alpha=0.7)
    ax_img.set_xlabel("X [mm]")
    ax_img.set_ylabel("Y [mm]")
    plt.colorbar(im, ax=ax_img, shrink=0.85, label=UNITS[mod])

    title = ax_img.set_title("", fontsize=11)

    ax_soc.plot(t_h, meta.soc_pct.values, "-", color="#222", lw=1)
    soc_dot, = ax_soc.plot([], [], "o", color="tab:orange", ms=8)
    ax_soc.set_ylabel("SOC [%]")
    ax_soc.grid(alpha=0.3)

    ax_v.plot(t_h, meta.voltage_at_scan.values, "-", color="tab:red", lw=1)
    v_dot, = ax_v.plot([], [], "o", color="tab:orange", ms=8)
    ax_v.set_ylabel("V [V]")
    ax_v.grid(alpha=0.3)

    ax_t.plot(t_h, meta.temp_at_scan.values, "-", color="tab:purple", lw=1)
    t_dot, = ax_t.plot([], [], "o", color="tab:orange", ms=8)
    ax_t.set_ylabel("T [°C]")
    ax_t.set_xlabel("time [h]")
    ax_t.grid(alpha=0.3)

    def update(i):
        im.set_data(arr[i])
        row = meta.iloc[i]
        tag = row.step_tag
        col = COLOR_BY_TAG.get(tag, "black")
        title.set_text(
            f"{mod}  ·  r{int(row.run_idx):03d}  ·  t={t_h[i]:6.2f} h\n"
            f"SOC={row.soc_pct:+6.1f}%   V={row.voltage_at_scan:.3f} V   T={row.temp_at_scan:.2f} °C   [{tag}]"
        )
        title.set_color(col)
        soc_dot.set_data([t_h[i]], [row.soc_pct])
        v_dot.set_data([t_h[i]], [row.voltage_at_scan])
        t_dot.set_data([t_h[i]], [row.temp_at_scan])
        return im, soc_dot, v_dot, t_dot, title

    anim = FuncAnimation(fig, update, frames=n, interval=120, blit=False)
    return fig, anim


def main() -> None:
    stack = load()
    for mod in ["tof", "amplitude", "energy"]:
        fig, anim = _build_anim(stack, mod)
        out_path = OUT / f"longrun_{mod}.mp4"
        try:
            anim.save(out_path, fps=8, dpi=120, writer="ffmpeg")
            print(f"wrote {out_path}")
        except Exception as e:
            gif_path = out_path.with_suffix(".gif")
            anim.save(gif_path, fps=8, dpi=100, writer="pillow")
            print(f"ffmpeg unavailable ({e}); wrote {gif_path} instead")
        plt.close(fig)


if __name__ == "__main__":
    main()
