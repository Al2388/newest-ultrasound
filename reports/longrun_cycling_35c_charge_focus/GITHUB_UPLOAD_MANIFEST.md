# 35C Long-Run Upload Manifest

This upload contains the GitHub-friendly 35C charge-focus analysis package.

## Included data

- `_cache/stack.npz`  
  Compact array bundle used by the analysis scripts. It contains the aligned
  `tof`, `amplitude`, and `energy` C-scan stacks, spatial axes, ROI mask, and
  related numeric arrays.
- `_cache/meta.csv`  
  Per-scan metadata, including timestamps, voltage-at-scan, line temperature
  summaries, run indices, and step tags.
- `data/raw/cycler/LFP860_35degrees.002.txt`  
  Raw cycler log for the 35C run.
- `06_animation/longrun_35c_gifs.zip`  
  Zip containing `longrun_35c_tof.gif`, `longrun_35c_energy.gif`,
  `longrun_35c_amplitude.gif`, and `longrun_35c_combined.gif`.

## Included code

- `scripts/charge_focus/plot_35c_animation.py`
- `scripts/charge_focus/plot_35c_animation_combined.py`
- `scripts/charge_focus/make_35c_scan_artifact_review.py`
- `scripts/charge_focus/fix_fullfield_display_artifact_rows.py`

## Not included

The original raw C-scan acquisition directories under
`data/raw/cscan/cscan_longrun_cycling_35c_2026-05-30_19-47-06_*` are not
committed. They are approximately 0.5 GB per scan and roughly 85 GB total for
the campaign, which is not suitable for a normal GitHub repository commit.

The committed `stack.npz` is the compact analysis-ready representation of the
raw amplitude, ToF, and energy maps used to generate the plots and GIFs.
