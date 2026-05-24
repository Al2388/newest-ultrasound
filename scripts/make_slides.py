"""
Assemble the SOC/temperature feature analysis into a 4-slide deck (PDF + PNGs).

Each slide = title + an existing figure + bullet takeaways, laid out on a
16:9 canvas. Reads the PNGs produced by feature_repeatability.py and
tof_temp_compensation.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REP = Path("reports/experiments/repeatability")
OUT = REP / "slides"
OUT.mkdir(exist_ok=True)

SLIDES = [
    {
        "title": "Acoustic features for SOC: separating SOC from temperature",
        "image": REP / "1_soc_temp_collinearity.png",
        "bullets": [
            "Two repeat runs of the identical LFP protocol (18-5, 21-5); full charge + full discharge.",
            "Within one run, SOC and temperature are collinear (r -0.84 to -0.92) - a single run cannot separate them.",
            "The two runs sat ~1 C apart at every SOC, so comparing them AT MATCHED SOC isolates temperature sensitivity.",
            "This cross-session design is the rigorous disentangler used throughout.",
        ],
    },
    {
        "title": "Which features track SOC, not temperature",
        "image": REP / "2_selectivity_map.png",
        "bullets": [
            "Y = SOC sensitivity, X = temperature sensitivity (partial R2). Above the diagonal = SOC dominates.",
            "Strongest SOC features (top of chart): early_p2p_v (0.73), centroid_time_us (0.71),",
            "  h5_tof_us_absolute (0.70), early_late_p2p_ratio (0.69), late_energy (0.63), energy_centroid_time_us (0.60).",
            "All sit well above the diagonal -> they carry far more SOC than temperature information.",
            "Ranked shortlist with the temperature split is on the next slide.",
        ],
    },
    {
        "title": "Shortlist: SOC sensitivity vs temperature sensitivity",
        "image": REP / "7_soc_vs_temp_sensitivity_bars.png",
        "bullets": [
            "Same metric as the previous slide, ranked: blue = SOC, red = temperature (partial R2).",
            "Filtered to features that actually carry SOC signal (SOC partial-R2 > 0.25), so 'selective but flat' features are excluded.",
            "Two physical groups (~0.9 correlated within each): TIMING (ToF / centroid / late-early energy) and AMPLITUDE (envelope_peak_v).",
            "Recommended representatives: energy_centroid_time_us (robust timing, tiny temp) + envelope_peak_v (independent amplitude).",
            "Raw ToF is SOC-rich but the most temperature-exposed of the strong features.",
        ],
    },
    {
        "title": "Repeatability: the protocol reproduces between sessions",
        "image": REP / "3_repeatability_top_features.png",
        "bullets": [
            "Top features overlay well between 18-5 (solid) and 21-5 (dashed) on charge and discharge.",
            "SOC-driven span is 3.6-4.9x the offset-removed run-to-run RMS (=> ~4-5 distinguishable SOC levels).",
            "Charge vs discharge hysteresis is real (LFP) and compared within-branch, not counted as error.",
        ],
    },
    {
        "title": "Temperature compensation of ToF helps only ~13%",
        "image": REP / "5_tof_temp_compensation.png",
        "bullets": [
            "Clean temperature coefficient ~ -0.05 us/C (pooled and optimal estimates agree).",
            "Best-case correction cuts between-session ToF RMS only 105 -> 91 ns (13%).",
            "=> ~87% of run-to-run ToF spread is NOT temperature (other drift: relaxation, coupling, electronics).",
            "Conclusion: don't rely on temp-correcting ToF; prefer intrinsically SOC-selective, temp-robust features.",
        ],
    },
    {
        "title": "Appendix: 18-5 full-cycle overview",
        "image": REP / "8_cycle_overview_18-5.png",
        "bullets": [
            "All streams on a shared time axis for one repeat run.",
            "Voltage dips = initial partial discharge; SOC ramps 0->100->0 then a partial top-up.",
            "ToF peaks near full charge (~13 h); temperature drifts ~1 C over the run.",
            "Amplitude and energy rise toward full charge, consistent with the SOC trends.",
        ],
    },
]


def render(slide, ax_fig):
    fig = plt.figure(figsize=(13.33, 7.5), dpi=150)
    fig.suptitle(slide["title"], fontsize=17, fontweight="bold", x=0.5, y=0.96)
    # image on the left/top
    ax_img = fig.add_axes((0.04, 0.08, 0.62, 0.80))
    ax_img.axis("off")
    if slide["image"].exists():
        ax_img.imshow(mpimg.imread(slide["image"]))
    else:
        ax_img.text(0.5, 0.5, f"missing:\n{slide['image'].name}", ha="center")
    # bullets on the right
    ax_txt = fig.add_axes((0.68, 0.08, 0.30, 0.80))
    ax_txt.axis("off")
    y = 0.92
    for b in slide["bullets"]:
        lead = "•  " if not b.startswith("  ") else "    "
        ax_txt.text(0.0, y, lead + b.strip(), fontsize=10.5, va="top", wrap=True,
                    transform=ax_txt.transAxes)
        y -= 0.10 + 0.045 * (len(b) // 38)
    return fig


def main():
    pdf_path = OUT / "soc_feature_analysis_deck.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, slide in enumerate(SLIDES, 1):
            fig = render(slide, None)
            pdf.savefig(fig)
            fig.savefig(OUT / f"slide_{i}.png")
            plt.close(fig)
    print("Wrote", pdf_path)
    for i in range(1, len(SLIDES) + 1):
        print("Wrote", OUT / f"slide_{i}.png")


if __name__ == "__main__":
    main()
