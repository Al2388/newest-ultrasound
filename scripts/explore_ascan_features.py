"""
Explore ultrasound A-scan features against cycler SOC and temperature.

This is intentionally broad rather than final-model clever. It extracts a wide
set of scalar features from each averaged A-scan waveform, aligns them with the
Maccor cycler export and temperature log, then scores each feature for:

  - SOC sensitivity
  - temperature sensitivity
  - partial R2 from SOC/branch after temperature
  - partial R2 from temperature after SOC/branch
  - simple rest-window noise

The goal is to discover which ultrasound observables are worth defending in a
report, not to overfit a single cycle.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class AcousticData:
    unix_s: np.ndarray
    elapsed_s: np.ndarray
    waveforms: h5py.Dataset
    attrs: dict
    existing: dict[str, np.ndarray]


def envelope_hilbert_2d(x: np.ndarray) -> np.ndarray:
    """Pure-NumPy analytic-signal envelope for rows of real waveforms."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    H = np.zeros(n, dtype=np.float32)
    if n % 2 == 0:
        H[0] = 1.0
        H[1 : n // 2] = 2.0
        H[n // 2] = 1.0
    else:
        H[0] = 1.0
        H[1 : (n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(X * H, axis=-1)).astype(np.float32, copy=False)


def parabolic_peak(y: np.ndarray, k: int) -> float:
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return float(k)
    return float(k) + max(-1.0, min(1.0, 0.5 * (y0 - y2) / denom))


def ncc_shift_and_corr(scan: np.ndarray, ref: np.ndarray, dt_s: float) -> tuple[float, float]:
    s = scan - np.mean(scan)
    r = ref - np.mean(ref)
    norm = float(np.sqrt(np.dot(s, s) * np.dot(r, r)))
    if norm <= 1e-12:
        return np.nan, np.nan
    corr = np.correlate(s, r, mode="full") / norm
    k = int(np.argmax(corr))
    center = len(ref) - 1
    shift_s = (parabolic_peak(corr, k) - center) * dt_s
    return float(shift_s), float(corr[k])


def first_threshold_time(x: np.ndarray, threshold: float) -> float:
    idx = np.flatnonzero(np.abs(x) >= threshold)
    return float(idx[0]) if idx.size else np.nan


def initial_rise_time(env: np.ndarray, frac_low: float = 0.1, frac_high: float = 0.9) -> float:
    peak = float(np.nanmax(env))
    if not np.isfinite(peak) or peak <= 1e-12:
        return np.nan
    peak_idx = int(np.nanargmax(env))
    before = env[: peak_idx + 1]
    lo = np.flatnonzero(before >= frac_low * peak)
    hi = np.flatnonzero(before >= frac_high * peak)
    if not lo.size or not hi.size:
        return np.nan
    return float(hi[0] - lo[0])


def spectral_features(
    x: np.ndarray,
    fs_hz: float,
    bands_hz: list[tuple[float, float]],
    rolloff_fraction: float = 0.85,
) -> dict[str, float]:
    w = np.hanning(x.size).astype(np.float32)
    y = (x - np.mean(x)) * w
    spec = np.abs(np.fft.rfft(y))
    power = spec * spec
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs_hz)
    total = float(np.sum(power))
    if total <= 1e-20:
        out = {
            "fft_peak_mhz": np.nan,
            "spectral_centroid_mhz": np.nan,
            "spectral_bandwidth_mhz": np.nan,
            "spectral_rolloff85_mhz": np.nan,
            "spectral_entropy": np.nan,
            "dominant_band_fraction": np.nan,
        }
        for i, _ in enumerate(bands_hz):
            out[f"bandpower_{i}"] = np.nan
        return out

    p = power / total
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    csum = np.cumsum(power)
    roll_idx = int(np.searchsorted(csum, rolloff_fraction * total))
    entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])) / math.log2(len(p)))
    bandpowers = []
    for lo, hi in bands_hz:
        m = (freqs >= lo) & (freqs < hi)
        bandpowers.append(float(np.sum(power[m]) / total))
    peak_i = int(np.argmax(power[1:]) + 1) if len(power) > 1 else 0
    out = {
        "fft_peak_mhz": float(freqs[peak_i] / 1e6),
        "spectral_centroid_mhz": centroid / 1e6,
        "spectral_bandwidth_mhz": bandwidth / 1e6,
        "spectral_rolloff85_mhz": float(freqs[min(roll_idx, len(freqs) - 1)] / 1e6),
        "spectral_entropy": entropy,
        "dominant_band_fraction": float(np.nanmax(bandpowers)) if bandpowers else np.nan,
    }
    for i, bp in enumerate(bandpowers):
        out[f"bandpower_{i}"] = bp
    return out


def stft_band_energy(
    x: np.ndarray,
    fs_hz: float,
    bands_hz: list[tuple[float, float]],
    nperseg: int = 128,
    hop: int = 64,
) -> dict[str, float]:
    if x.size < nperseg:
        return {f"stft_bandpower_{i}": np.nan for i, _ in enumerate(bands_hz)}
    accum = np.zeros(len(bands_hz), dtype=np.float64)
    total = 0.0
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs_hz)
    window = np.hanning(nperseg).astype(np.float32)
    for start in range(0, x.size - nperseg + 1, hop):
        seg = (x[start : start + nperseg] - np.mean(x[start : start + nperseg])) * window
        power = np.abs(np.fft.rfft(seg)) ** 2
        total += float(np.sum(power))
        for i, (lo, hi) in enumerate(bands_hz):
            accum[i] += float(np.sum(power[(freqs >= lo) & (freqs < hi)]))
    if total <= 1e-20:
        return {f"stft_bandpower_{i}": np.nan for i, _ in enumerate(bands_hz)}
    return {f"stft_bandpower_{i}": float(v / total) for i, v in enumerate(accum)}


def haar_packet_energy(x: np.ndarray, levels: int = 3) -> dict[str, float]:
    """Small wavelet-packet-like Haar decomposition, dependency-free."""
    coeffs = [x.astype(np.float64) - np.mean(x)]
    for _ in range(levels):
        nxt = []
        for c in coeffs:
            if len(c) % 2:
                c = c[:-1]
            a = (c[0::2] + c[1::2]) / np.sqrt(2.0)
            d = (c[0::2] - c[1::2]) / np.sqrt(2.0)
            nxt.extend([a, d])
        coeffs = nxt
    energies = np.asarray([float(np.dot(c, c)) for c in coeffs], dtype=np.float64)
    total = float(np.sum(energies))
    if total <= 1e-20:
        return {f"haar_packet_e{i}": np.nan for i in range(len(energies))}
    return {f"haar_packet_e{i}": float(e / total) for i, e in enumerate(energies)}


def pca_scores(X: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Xc = np.nan_to_num(Xc, nan=0.0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ vt[:n_components].T
    explained = (s[:n_components] ** 2) / np.sum(s**2)
    return scores, explained


def read_acoustic(path: Path) -> tuple[h5py.File, AcousticData]:
    f = h5py.File(path, "r")
    ts = f["timestamps"][:]
    existing = {}
    for name in [
        "tof_us",
        "tof_us_absolute",
        "tof_us_envelope",
        "amplitude",
        "amplitude_envelope",
        "energy",
        "clip_fraction",
        "raw_clip_fraction",
        "tracking_corr",
        "tracking_lag_samples",
        "n_averaged",
        "n_rejected",
        "waveform_max_v",
        "waveform_min_v",
    ]:
        if name in f:
            existing[name] = f[name][:]
    return f, AcousticData(
        unix_s=ts,
        elapsed_s=ts - ts[0],
        waveforms=f["waveforms"],
        attrs=dict(f.attrs),
        existing=existing,
    )


def read_temperature(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                h, m, s = row[0].strip('"').split(":")
                elapsed = int(h) * 3600 + int(m) * 60 + int(s)
                rows.append((elapsed, float(row[1])))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["temp_elapsed_s", "temperature_c"])
    df["temp_elapsed_s"] -= df["temp_elapsed_s"].iloc[0]
    return df


def read_maccor_txt(path: Path, q_nominal_ah: float, tz_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", skiprows=6, engine="python").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    for col in ["Rec", "Cycle P", "Cycle C", "Step", "Capacity", "Energy", "Current", "Voltage"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    local_tz = ZoneInfo(tz_name)
    dt_local = pd.to_datetime(df["DPT Time"], dayfirst=True, errors="coerce")
    # Pandas keeps this object-ish on Windows/older versions; list conversion is robust.
    unix_s = np.asarray(
        [
            x.replace(tzinfo=local_tz).astimezone(ZoneInfo("UTC")).timestamp()
            if pd.notna(x)
            else np.nan
            for x in dt_local
        ],
        dtype=np.float64,
    )
    df["cycler_unix_s"] = unix_s
    df["cycler_elapsed_s"] = unix_s - unix_s[np.isfinite(unix_s)][0]
    mode = df["MD"].astype(str).str.strip()
    signed_current = np.zeros(len(df), dtype=np.float64)
    signed_current[mode == "C"] = pd.to_numeric(df.loc[mode == "C", "Current"], errors="coerce")
    signed_current[mode == "D"] = -pd.to_numeric(df.loc[mode == "D", "Current"], errors="coerce")
    df["signed_current_a"] = np.nan_to_num(signed_current, nan=0.0)
    dt_h = np.diff(df["cycler_unix_s"].to_numpy(dtype=np.float64)) / 3600.0
    i_mid = 0.5 * (df["signed_current_a"].to_numpy()[:-1] + df["signed_current_a"].to_numpy()[1:])
    valid = np.isfinite(dt_h) & (dt_h >= 0) & (dt_h < 0.1)
    q = np.zeros(len(df), dtype=np.float64)
    q[1:] = np.cumsum(np.where(valid, i_mid * dt_h, 0.0))
    df["relative_q_ah"] = q
    # Anchor SOC to the first observed low-SOC point, usually the end of step 2.
    q_zero = float(np.nanmin(q))
    df["soc_pct"] = (q - q_zero) / q_nominal_ah * 100.0
    df["soc_pct_clipped"] = df["soc_pct"].clip(0.0, 100.0)
    df["branch"] = np.select(
        [mode == "C", mode == "D", mode == "R"],
        ["charge", "discharge", "rest"],
        default="other",
    )
    return df


def nearest_indices(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(source_t, target_t), 0, len(source_t) - 1)
    left = np.clip(idx - 1, 0, len(source_t) - 1)
    return np.where(np.abs(source_t[left] - target_t) <= np.abs(source_t[idx] - target_t), left, idx)


def sample_acoustic_indices(ac: AcousticData, target_period_s: float) -> np.ndarray:
    if target_period_s <= 0:
        return np.arange(len(ac.unix_s), dtype=int)
    grid = np.arange(ac.unix_s[0], ac.unix_s[-1] + 0.1, target_period_s)
    return np.unique(nearest_indices(ac.unix_s, grid))


def extract_features(ac: AcousticData, indices: np.ndarray, ref_index: int) -> pd.DataFrame:
    fs = float(ac.attrs.get("fs_hz", 20_000_000.0))
    dt_s = 1.0 / fs
    gate_start_us = float(ac.attrs.get("gate_us_start", 0.0))
    n = ac.waveforms.shape[1]
    sample_us = gate_start_us + np.arange(n) * dt_s * 1e6
    bands = [(0.2e6, 0.8e6), (0.8e6, 1.5e6), (1.5e6, 3.0e6), (3.0e6, 5.0e6), (5.0e6, 9.5e6)]
    ref = ac.waveforms[ref_index]
    ref_norm = ref - np.mean(ref)
    ref_norm = ref_norm / max(float(np.linalg.norm(ref_norm)), 1e-12)

    rows = []
    # Keep PCA on a regular subset of the selected waveforms.
    wf_for_pca = ac.waveforms[indices[: min(len(indices), 12000)]]
    pca_subset_scores, pca_explained = pca_scores(wf_for_pca, n_components=5)
    pca_basis = None
    if len(wf_for_pca) >= 5:
        Xc = wf_for_pca - np.nanmean(wf_for_pca, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(np.nan_to_num(Xc, nan=0.0), full_matrices=False)
        pca_basis = vt[:5]
        pca_mean = np.nanmean(wf_for_pca, axis=0)
    else:
        pca_mean = np.zeros(n)

    for local_i, idx in enumerate(indices):
        w = ac.waveforms[idx].astype(np.float64)
        w_dc = w - np.mean(w)
        env = envelope_hilbert_2d(w_dc[None, :])[0].astype(np.float64)
        abs_w = np.abs(w_dc)
        env_peak_idx = int(np.nanargmax(env))
        abs_peak_idx = int(np.nanargmax(abs_w))
        centroid_idx = float(np.sum(np.arange(n) * abs_w) / max(np.sum(abs_w), 1e-12))
        energy_cum = np.cumsum(w_dc * w_dc)
        total_energy = float(energy_cum[-1])
        energy_centroid_idx = float(np.sum(np.arange(n) * (w_dc * w_dc)) / max(total_energy, 1e-12))
        early = w_dc[: n // 2]
        late = w_dc[n // 2 :]
        early_e = float(np.dot(early, early))
        late_e = float(np.dot(late, late))
        shift_s, corr = ncc_shift_and_corr(w_dc, ref, dt_s)
        norm_w = w_dc / max(float(np.linalg.norm(w_dc)), 1e-12)
        mse_ref = float(np.mean((norm_w - ref_norm) ** 2))
        rmse_ref = float(np.sqrt(mse_ref))
        row = {
            "ascan_index": int(idx),
            "ascan_unix_s": float(ac.unix_s[idx]),
            "ascan_elapsed_h": float(ac.elapsed_s[idx] / 3600.0),
            "p2p_v": float(np.nanmax(w) - np.nanmin(w)),
            "max_abs_v": float(np.nanmax(abs_w)),
            "mean_abs_v": float(np.nanmean(abs_w)),
            "rms_v": float(np.sqrt(np.nanmean(w_dc * w_dc))),
            "crest_factor": float(np.nanmax(abs_w) / max(np.sqrt(np.nanmean(w_dc * w_dc)), 1e-12)),
            "shape_factor": float(np.sqrt(np.nanmean(w_dc * w_dc)) / max(np.nanmean(abs_w), 1e-12)),
            "skewness": float(np.nanmean((w_dc / max(np.nanstd(w_dc), 1e-12)) ** 3)),
            "kurtosis": float(np.nanmean((w_dc / max(np.nanstd(w_dc), 1e-12)) ** 4)),
            "ringing_count": int(np.sum(np.diff(np.signbit(w_dc)) != 0)),
            "envelope_peak_v": float(np.nanmax(env)),
            "envelope_integral": float(np.trapz(env, sample_us)),
            "envelope_peak_tof_us": float(sample_us[env_peak_idx]),
            "abs_peak_tof_us": float(sample_us[abs_peak_idx]),
            "threshold10_tof_us": float(gate_start_us + first_threshold_time(w_dc, 0.10 * np.nanmax(abs_w)) * dt_s * 1e6),
            "threshold50_tof_us": float(gate_start_us + first_threshold_time(w_dc, 0.50 * np.nanmax(abs_w)) * dt_s * 1e6),
            "centroid_time_us": float(gate_start_us + centroid_idx * dt_s * 1e6),
            "energy_centroid_time_us": float(gate_start_us + energy_centroid_idx * dt_s * 1e6),
            "initial_rise_time_us": float(initial_rise_time(env) * dt_s * 1e6),
            "ncc_shift_us": float(shift_s * 1e6),
            "ncc_corr_ref": corr,
            "energy_total": total_energy,
            "log_energy_total": float(np.log(max(total_energy, 1e-20))),
            "early_energy": early_e,
            "late_energy": late_e,
            "early_late_energy_ratio": float(early_e / max(late_e, 1e-20)),
            "early_p2p_v": float(np.nanmax(early) - np.nanmin(early)),
            "late_p2p_v": float(np.nanmax(late) - np.nanmin(late)),
            "early_late_p2p_ratio": float((np.nanmax(early) - np.nanmin(early)) / max(np.nanmax(late) - np.nanmin(late), 1e-12)),
            "cosine_similarity_ref": float(np.dot(norm_w, ref_norm)),
            "mse_to_ref": mse_ref,
            "rmse_to_ref": rmse_ref,
            "dtw_proxy_l1_to_ref": float(np.mean(np.abs(norm_w - ref_norm))),
        }
        row.update(spectral_features(w_dc, fs, bands))
        row.update(stft_band_energy(w_dc, fs, bands))
        row.update(haar_packet_energy(w_dc, levels=3))
        if pca_basis is not None:
            scores = (np.nan_to_num(w - pca_mean, nan=0.0) @ pca_basis.T)
            for j, score in enumerate(scores):
                row[f"pca_score_{j + 1}"] = float(score)
                row[f"pca_explained_{j + 1}"] = float(pca_explained[j]) if j < len(pca_explained) else np.nan
        for name, arr in ac.existing.items():
            row[f"h5_{name}"] = float(arr[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def align_context(features: pd.DataFrame, cycler: pd.DataFrame, temp: pd.DataFrame, temp_start_unix_s: float) -> pd.DataFrame:
    out = features.copy()
    cidx = nearest_indices(cycler["cycler_unix_s"].to_numpy(dtype=np.float64), out["ascan_unix_s"].to_numpy(dtype=np.float64))
    for col in ["cycler_elapsed_s", "Step", "Capacity", "Energy", "Current", "Voltage", "MD", "signed_current_a", "relative_q_ah", "soc_pct", "soc_pct_clipped", "branch"]:
        out[col] = cycler.iloc[cidx][col].to_numpy()
    temp_unix = temp_start_unix_s + temp["temp_elapsed_s"].to_numpy(dtype=np.float64)
    out["temperature_c"] = np.interp(out["ascan_unix_s"], temp_unix, temp["temperature_c"], left=temp["temperature_c"].iloc[0], right=temp["temperature_c"].iloc[-1])
    out["time_h"] = (out["ascan_unix_s"] - out["ascan_unix_s"].iloc[0]) / 3600.0
    return out


def design_matrices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    soc = df["soc_pct_clipped"].to_numpy(dtype=float) / 100.0
    temp = df["temperature_c"].to_numpy(dtype=float)
    time_h = df["time_h"].to_numpy(dtype=float)
    charge = (df["branch"].to_numpy() == "charge").astype(float)
    discharge = (df["branch"].to_numpy() == "discharge").astype(float)
    rest = (df["branch"].to_numpy() == "rest").astype(float)
    current = df["signed_current_a"].to_numpy(dtype=float)
    return {
        "temp": np.column_stack([np.ones(len(df)), temp]),
        "soc_state": np.column_stack([np.ones(len(df)), soc, soc**2, soc**3, charge, discharge, rest, soc * charge, soc * discharge]),
        "full": np.column_stack([np.ones(len(df)), soc, soc**2, soc**3, temp, charge, discharge, rest, soc * charge, soc * discharge, current, time_h]),
    }


def fit_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    if int(mask.sum()) < X.shape[1] + 2:
        return np.nan, np.nan
    coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    pred = X[mask] @ coef
    sse = float(np.sum((y[mask] - pred) ** 2))
    sst = float(np.sum((y[mask] - np.mean(y[mask])) ** 2))
    return (1.0 - sse / sst if sst > 0 else np.nan), sse


def score_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = design_matrices(df)
    rows = []
    rest = df["branch"].eq("rest").to_numpy()
    charge = df["branch"].eq("charge").to_numpy()
    discharge = df["branch"].eq("discharge").to_numpy()
    for col in feature_cols:
        y = df[col].to_numpy(dtype=float)
        r2_temp, sse_temp = fit_r2(y, X["temp"])
        r2_soc, sse_soc = fit_r2(y, X["soc_state"])
        r2_full, sse_full = fit_r2(y, X["full"])
        y_span = float(np.nanpercentile(y, 99) - np.nanpercentile(y, 1))
        noise = float(np.nanmedian(np.abs(np.diff(y[rest])))) if np.sum(rest) > 3 else np.nan
        partial_soc_after_temp = (sse_temp - sse_full) / sse_temp if np.isfinite(sse_temp) and sse_temp > 0 else np.nan
        partial_temp_after_soc = (sse_soc - sse_full) / sse_soc if np.isfinite(sse_soc) and sse_soc > 0 else np.nan
        # Linear local sensitivities are descriptive; full relationship may be nonlinear.
        valid = np.isfinite(y) & np.isfinite(df["soc_pct_clipped"]) & np.isfinite(df["temperature_c"])
        soc_slope = np.nan
        temp_slope = np.nan
        if np.sum(valid) > 10:
            A = np.column_stack([np.ones(np.sum(valid)), df.loc[valid, "soc_pct_clipped"], df.loc[valid, "temperature_c"]])
            coef, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
            soc_slope = float(coef[1])
            temp_slope = float(coef[2])
        charge_slope, charge_corr = branch_slope_corr(df, y, charge)
        discharge_slope, discharge_corr = branch_slope_corr(df, y, discharge)
        rest_temp_slope, rest_temp_corr = temp_slope_corr(df, y, rest)
        rows.append(
            {
                "feature": col,
                "span_1_99": y_span,
                "rest_diff_noise": noise,
                "span_to_noise": y_span / noise if np.isfinite(noise) and noise > 0 else np.nan,
                "linear_sensitivity_per_soc_pct": soc_slope,
                "linear_sensitivity_per_c": temp_slope,
                "r2_temp_only": r2_temp,
                "r2_soc_branch_only": r2_soc,
                "r2_full": r2_full,
                "partial_r2_soc_after_temp": partial_soc_after_temp,
                "partial_r2_temp_after_soc": partial_temp_after_soc,
                "charge_slope_per_soc_pct": charge_slope,
                "charge_corr_soc": charge_corr,
                "discharge_slope_per_soc_pct": discharge_slope,
                "discharge_corr_soc": discharge_corr,
                "rest_slope_per_c": rest_temp_slope,
                "rest_corr_temp": rest_temp_corr,
            }
        )
    score = pd.DataFrame(rows)
    score["priority_score"] = (
        score["partial_r2_soc_after_temp"].fillna(0)
        * np.log10(score["span_to_noise"].replace([np.inf, -np.inf], np.nan).fillna(1).clip(lower=1))
        - 0.5 * score["partial_r2_temp_after_soc"].fillna(0)
    )
    return score.sort_values("priority_score", ascending=False)


def branch_slope_corr(df: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(y) & np.isfinite(df["soc_pct_clipped"].to_numpy(dtype=float))
    if int(np.sum(valid)) < 5:
        return np.nan, np.nan
    x = df.loc[valid, "soc_pct_clipped"].to_numpy(dtype=float)
    yy = y[valid]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    corr = np.corrcoef(x, yy)[0, 1] if np.nanstd(x) > 0 and np.nanstd(yy) > 0 else np.nan
    return float(coef[1]), float(corr)


def temp_slope_corr(df: pd.DataFrame, y: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask & np.isfinite(y) & np.isfinite(df["temperature_c"].to_numpy(dtype=float))
    if int(np.sum(valid)) < 5:
        return np.nan, np.nan
    x = df.loc[valid, "temperature_c"].to_numpy(dtype=float)
    yy = y[valid]
    A = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    corr = np.corrcoef(x, yy)[0, 1] if np.nanstd(x) > 0 and np.nanstd(yy) > 0 else np.nan
    return float(coef[1]), float(corr)


def plot_overview(df: pd.DataFrame, out_dir: Path, top_features: list[str]) -> None:
    fig, axs = plt.subplots(4, 1, figsize=(13, 9), dpi=150, sharex=True)
    axs[0].plot(df["time_h"], df["Voltage"], color="#1d4ed8", lw=0.9)
    axs[0].set_ylabel("Voltage (V)")
    ax0b = axs[0].twinx()
    ax0b.plot(df["time_h"], df["signed_current_a"], color="#6b7280", lw=0.7, alpha=0.7)
    ax0b.set_ylabel("Current (A)")
    axs[1].plot(df["time_h"], df["soc_pct_clipped"], color="#047857", lw=0.9)
    axs[1].set_ylabel("SOC (%)")
    ax1b = axs[1].twinx()
    ax1b.plot(df["time_h"], df["temperature_c"], color="#c2410c", lw=0.7, alpha=0.8)
    ax1b.set_ylabel("Temp (C)")
    for feature in top_features[:2]:
        axs[2].plot(df["time_h"], df[feature], lw=0.8, label=feature)
    axs[2].legend(loc="best", fontsize=8)
    axs[2].set_ylabel("Top features")
    for feature in top_features[2:5]:
        axs[3].plot(df["time_h"], df[feature], lw=0.8, label=feature)
    axs[3].legend(loc="best", fontsize=8)
    axs[3].set_ylabel("More features")
    axs[3].set_xlabel("Elapsed time (h)")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_exploration_overview.png")
    plt.close(fig)

    for feature in top_features[:12]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=150)
        for branch, color in [("charge", "#2563eb"), ("discharge", "#dc2626"), ("rest", "#6b7280")]:
            m = df["branch"].eq(branch)
            ax.scatter(df.loc[m, "soc_pct_clipped"], df.loc[m, feature], s=8, alpha=0.55, label=branch, color=color)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel(feature)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"soc_scatter_{safe_name(feature)}.png")
        plt.close(fig)


def plot_score_summary(score: pd.DataFrame, out_dir: Path) -> None:
    top = score.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.barh(top["feature"], top["partial_r2_soc_after_temp"], color="#2563eb", label="SOC after temp")
    ax.barh(top["feature"], -top["partial_r2_temp_after_soc"], color="#dc2626", label="Temp after SOC")
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Partial R2; temperature bars plotted negative")
    ax.set_title("Feature separation score")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_partial_r2_bar.png")
    plt.close(fig)

    cols = [
        "partial_r2_soc_after_temp",
        "partial_r2_temp_after_soc",
        "span_to_noise",
        "charge_corr_soc",
        "discharge_corr_soc",
        "rest_corr_temp",
    ]
    heat = score.head(25).set_index("feature")[cols].copy()
    heat["span_to_noise"] = np.log10(heat["span_to_noise"].replace([np.inf, -np.inf], np.nan).clip(lower=1))
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["SOC|T", "T|SOC", "log SNR", "charge r", "disch r", "rest T r"], rotation=35, ha="right")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Top feature diagnostic matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_score_matrix.png")
    plt.close(fig)


def plot_branch_panels(df: pd.DataFrame, out_dir: Path, top_features: list[str]) -> None:
    for feature in top_features[:8]:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
        for ax, branch, color in zip(
            axs,
            ["charge", "discharge", "rest"],
            ["#2563eb", "#dc2626", "#6b7280"],
        ):
            m = df["branch"].eq(branch)
            if branch == "rest":
                x = df.loc[m, "temperature_c"]
                ax.set_xlabel("Temperature (C)")
            else:
                x = df.loc[m, "soc_pct_clipped"]
                ax.set_xlabel("SOC (%)")
            y = df.loc[m, feature]
            ax.scatter(x, y, s=8, alpha=0.5, color=color)
            if len(y) > 3:
                valid = np.isfinite(x) & np.isfinite(y)
                if int(valid.sum()) > 3:
                    coef = np.polyfit(x[valid], y[valid], 1)
                    xx = np.linspace(float(np.nanmin(x[valid])), float(np.nanmax(x[valid])), 80)
                    ax.plot(xx, coef[0] * xx + coef[1], color="#111827", lw=1.0)
            ax.set_title(branch)
            ax.grid(alpha=0.25)
        axs[0].set_ylabel(feature)
        fig.suptitle(f"Branch behavior: {feature}", y=1.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"branch_panel_{safe_name(feature)}.png")
        plt.close(fig)


def robust_norm(series: pd.Series) -> np.ndarray:
    y = series.to_numpy(dtype=float)
    lo, hi = np.nanpercentile(y, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return np.full_like(y, np.nan, dtype=float)
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def plot_easy_inspection(df: pd.DataFrame, score: pd.DataFrame, out_dir: Path) -> None:
    """Make a small set of readable plots for human inspection."""
    readable = [
        "h5_tof_us",
        "early_energy",
        "late_energy",
        "early_late_energy_ratio",
        "spectral_centroid_mhz",
        "spectral_entropy",
    ]
    readable = [c for c in readable if c in df.columns]

    fig, axs = plt.subplots(4, 1, figsize=(13, 9), dpi=160, sharex=True)
    axs[0].plot(df["time_h"], df["Voltage"], color="#1d4ed8", lw=1.0, label="Voltage")
    axs[0].set_ylabel("Voltage (V)")
    ax0 = axs[0].twinx()
    ax0.step(df["time_h"], df["signed_current_a"], where="post", color="#6b7280", lw=0.8, alpha=0.8, label="Current")
    ax0.set_ylabel("Current (A)")
    axs[0].set_title("19-5 cycle context and ultrasound feature movement")

    axs[1].plot(df["time_h"], df["soc_pct_clipped"], color="#047857", lw=1.2, label="SOC")
    axs[1].set_ylabel("SOC (%)")
    ax1 = axs[1].twinx()
    ax1.plot(df["time_h"], df["temperature_c"], color="#c2410c", lw=0.8, alpha=0.8)
    ax1.set_ylabel("Temp (C)")

    for col in readable[:4]:
        axs[2].plot(df["time_h"], robust_norm(df[col]), lw=0.95, label=col)
    axs[2].set_ylabel("Normalized\n0-1")
    axs[2].legend(loc="best", fontsize=8, ncol=2)

    for col in readable[4:]:
        axs[3].plot(df["time_h"], robust_norm(df[col]), lw=0.95, label=col)
    axs[3].set_ylabel("Spectral\n0-1")
    axs[3].set_xlabel("Elapsed time (h)")
    axs[3].legend(loc="best", fontsize=8)

    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "easy_overview_normalized_features.png")
    plt.close(fig)

    # Compact feature-vs-SOC panels.
    panel_features = readable
    ncols = 3
    nrows = int(math.ceil(len(panel_features) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), dpi=160, squeeze=False)
    for ax, feature in zip(axs.ravel(), panel_features):
        for branch, color in [("charge", "#2563eb"), ("discharge", "#dc2626"), ("rest", "#6b7280")]:
            m = df["branch"].eq(branch)
            ax.scatter(df.loc[m, "soc_pct_clipped"], df.loc[m, feature], s=7, alpha=0.45, color=color, label=branch)
        ax.set_title(feature)
        ax.set_xlabel("SOC (%)")
        ax.grid(alpha=0.25)
    for ax in axs.ravel()[len(panel_features) :]:
        ax.axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "easy_feature_vs_soc_panels.png")
    plt.close(fig)

    # Bar chart of a simpler headline metric.
    simple = score.copy()
    simple["soc_minus_temp"] = simple["partial_r2_soc_after_temp"] - simple["partial_r2_temp_after_soc"]
    top = simple.sort_values("soc_minus_temp", ascending=False).head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    ax.barh(top["feature"], top["soc_minus_temp"], color="#047857")
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("Partial R2(SOC after temp) - Partial R2(temp after SOC)")
    ax.set_title("Features most SOC-distinct after temperature correction")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "easy_soc_distinct_feature_ranking.png")
    plt.close(fig)


def plot_spectral_soc(df: pd.DataFrame, score: pd.DataFrame, out_dir: Path) -> None:
    spectral_cols = [
        c
        for c in df.columns
        if c.startswith("bandpower_")
        or c.startswith("stft_bandpower_")
        or c.startswith("haar_packet_e")
        or c in {"fft_peak_mhz", "spectral_centroid_mhz", "spectral_bandwidth_mhz", "spectral_rolloff85_mhz", "spectral_entropy", "dominant_band_fraction"}
    ]
    spectral_cols = [c for c in spectral_cols if c in score["feature"].values]
    if not spectral_cols:
        return

    rows = []
    for c in spectral_cols:
        y = df[c].to_numpy(dtype=float)
        for branch in ["charge", "discharge"]:
            m = df["branch"].eq(branch).to_numpy() & np.isfinite(y) & np.isfinite(df["soc_pct_clipped"].to_numpy(dtype=float))
            if int(m.sum()) > 5 and np.nanstd(y[m]) > 0:
                r = float(np.corrcoef(df.loc[m, "soc_pct_clipped"].to_numpy(dtype=float), y[m])[0, 1])
            else:
                r = np.nan
            rows.append((c, branch, r))
    corr = pd.DataFrame(rows, columns=["feature", "branch", "corr"])
    pivot = corr.pivot(index="feature", columns="branch", values="corr").fillna(0.0)
    order = pivot.abs().max(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(7, max(5, 0.35 * len(pivot))), dpi=160)
    im = ax.imshow(pivot[["charge", "discharge"]].to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["charge r(SOC)", "discharge r(SOC)"])
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spectral feature correlation with SOC")
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_soc_correlation_heatmap.png")
    plt.close(fig)

    best = [c for c in pivot.index[:6] if c in df.columns]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), dpi=160, squeeze=False)
    for ax, feature in zip(axs.ravel(), best):
        for branch, color in [("charge", "#2563eb"), ("discharge", "#dc2626")]:
            m = df["branch"].eq(branch)
            ax.scatter(df.loc[m, "soc_pct_clipped"], df.loc[m, feature], s=7, alpha=0.45, color=color, label=branch)
        ax.set_title(feature)
        ax.set_xlabel("SOC (%)")
        ax.grid(alpha=0.25)
    for ax in axs.ravel()[len(best) :]:
        ax.axis("off")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "spectral_feature_vs_soc_panels.png")
    plt.close(fig)

    corr.to_csv(out_dir / "spectral_soc_correlations.csv", index=False)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def write_report(out_dir: Path, args: argparse.Namespace, df: pd.DataFrame, score: pd.DataFrame) -> None:
    top = score.head(20)
    lines = [
        "# A-scan Feature Exploration",
        "",
        "## Inputs",
        f"- H5: `{args.h5}`",
        f"- Cycler: `{args.cycler}`",
        f"- Temperature: `{args.temp}`",
        "",
        "## Data Coverage",
        f"- Rows in feature table: {len(df):,}",
        f"- Time span: {df['time_h'].min():.3f} to {df['time_h'].max():.3f} h",
        f"- SOC span, clipped: {df['soc_pct_clipped'].min():.2f} to {df['soc_pct_clipped'].max():.2f} %",
        f"- Temperature span: {df['temperature_c'].min():.3f} to {df['temperature_c'].max():.3f} C",
        "",
        "## Highest Priority Features",
        "",
        "| Feature | partial R2 SOC after temp | partial R2 temp after SOC | span/noise | linear /%SOC | linear /C |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['partial_r2_soc_after_temp']:.3f} | "
            f"{r['partial_r2_temp_after_soc']:.3f} | {r['span_to_noise']:.1f} | "
            f"{r['linear_sensitivity_per_soc_pct']:.4g} | {r['linear_sensitivity_per_c']:.4g} |"
        )
    lines += [
        "",
        "## Branch-Aware Checks",
        "",
        "| Feature | charge slope /%SOC | charge r | discharge slope /%SOC | discharge r | rest slope /C | rest temp r |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.head(12).iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['charge_slope_per_soc_pct']:.4g} | {r['charge_corr_soc']:.3f} | "
            f"{r['discharge_slope_per_soc_pct']:.4g} | {r['discharge_corr_soc']:.3f} | "
            f"{r['rest_slope_per_c']:.4g} | {r['rest_corr_temp']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation Notes",
        "- Partial R2 SOC after temp asks: once temperature is already in the model, how much remaining feature variation is explained by SOC/branch/state?",
        "- Partial R2 temp after SOC asks the reverse. A useful SOC feature should generally have the first value larger than the second.",
        "- Linear sensitivities are only quick descriptors. LFP is hysteretic and nonlinear, so use the SOC scatter plots before making claims.",
        "- Branch slopes are fitted separately because LFP charge and discharge are path-dependent. Opposite signs are not automatically a problem.",
        "- Rest-window noise is estimated from successive differences during rest periods; it is a practical noise proxy, not a metrology-grade uncertainty.",
    ]
    (out_dir / "FEATURE_EXPLORATION.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default=r"data/raw/ascan/ascan_session_2026-05-18_17-08-27/ascan_session_2026-05-18_17-08-27.h5")
    parser.add_argument("--cycler", default=r"data/raw/cycler/19-5 cycle.txt")
    parser.add_argument("--temp", default=r"data/raw/temperature/19-5 temp.csv")
    parser.add_argument("--out-dir", default=r"reports/experiments/19-5_feature_exploration")
    parser.add_argument("--q-nominal-ah", type=float, default=0.86)
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--sample-period-s", type=float, default=10.0)
    parser.add_argument("--ref-index", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_file, ac = read_acoustic(Path(args.h5))
    try:
        indices = sample_acoustic_indices(ac, args.sample_period_s)
        features = extract_features(ac, indices, args.ref_index)
    finally:
        h5_file.close()

    cycler = read_maccor_txt(Path(args.cycler), args.q_nominal_ah, args.timezone)
    temp = read_temperature(Path(args.temp))
    # Temperature logger appears to start with this cycling run; use A-scan start.
    aligned = align_context(features, cycler, temp, temp_start_unix_s=features["ascan_unix_s"].iloc[0])
    aligned.to_csv(out_dir / "ascan_feature_table.csv", index=False)

    context_cols = {
        "ascan_index",
        "ascan_unix_s",
        "ascan_elapsed_h",
        "cycler_elapsed_s",
        "Step",
        "Capacity",
        "Energy",
        "Current",
        "Voltage",
        "MD",
        "signed_current_a",
        "relative_q_ah",
        "soc_pct",
        "soc_pct_clipped",
        "branch",
        "temperature_c",
        "time_h",
    }
    feature_cols = [
        c
        for c in aligned.columns
        if c not in context_cols and pd.api.types.is_numeric_dtype(aligned[c])
    ]
    score = score_features(aligned, feature_cols)
    score.to_csv(out_dir / "feature_sensitivity_scores.csv", index=False)
    top_features = score["feature"].head(12).tolist()
    plot_overview(aligned, out_dir, top_features)
    plot_score_summary(score, out_dir)
    plot_branch_panels(aligned, out_dir, top_features)
    plot_easy_inspection(aligned, score, out_dir)
    plot_spectral_soc(aligned, score, out_dir)
    write_report(out_dir, args, aligned, score)
    print(f"Wrote {out_dir / 'ascan_feature_table.csv'}")
    print(f"Wrote {out_dir / 'feature_sensitivity_scores.csv'}")
    print(f"Wrote {out_dir / 'FEATURE_EXPLORATION.md'}")


if __name__ == "__main__":
    main()
