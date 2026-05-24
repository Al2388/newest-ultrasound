from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    out = Path("reports/experiments/19-5_feature_exploration")
    image_names = [
        "easy_overview_normalized_features.png",
        "easy_feature_vs_soc_panels.png",
        "spectral_soc_correlation_heatmap.png",
        "spectral_feature_vs_soc_panels.png",
        "easy_soc_distinct_feature_ranking.png",
        "feature_partial_r2_bar.png",
        "feature_score_matrix.png",
        "branch_panel_h5_tof_us.png",
        "branch_panel_early_energy.png",
        "branch_panel_late_energy.png",
    ]
    image_names = [name for name in image_names if (out / name).exists()]

    fig, axs = plt.subplots(len(image_names), 1, figsize=(14, 5 * len(image_names)), dpi=120)
    if len(image_names) == 1:
        axs = [axs]
    for ax, name in zip(axs, image_names):
        image = mpimg.imread(out / name)
        ax.imshow(image)
        ax.set_title(name, fontsize=14, pad=10)
        ax.axis("off")
    fig.tight_layout()
    contact_sheet = out / "OPEN_THIS_contact_sheet.png"
    fig.savefig(contact_sheet)
    plt.close(fig)

    html = [
        "<!doctype html>",
        '<html><head><meta charset="utf-8"><title>19-5 Feature Exploration</title>',
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#111827}",
        "h1{margin-bottom:4px}section{margin:28px 0;padding:18px;background:white;border:1px solid #e5e7eb}",
        "img{max-width:100%;height:auto;border:1px solid #e5e7eb}code{background:#eef2ff;padding:2px 4px}",
        "</style></head><body>",
        "<h1>19-5 Feature Exploration Gallery</h1>",
        "<p>Open this page in a browser. Images are linked relative to this HTML file.</p>",
    ]
    for name in image_names:
        html.append(f'<section><h2>{name}</h2><img src="{name}" alt="{name}"></section>')
    html.append("</body></html>")
    gallery = out / "OPEN_THIS_gallery.html"
    gallery.write_text("\n".join(html), encoding="utf-8")

    print(contact_sheet.resolve())
    print(gallery.resolve())


if __name__ == "__main__":
    main()
