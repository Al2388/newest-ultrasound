#!/usr/bin/env python
"""Run repeated C-scans through the web API and summarize the noise floor."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CFG = {
    "roi_w": 80.0,
    "roi_h": 72.0,
    "pitch": 0.5,
    "speed": 25.0,
    "accel_mode": "auto",
    "accel_mm_s2": 800.0,
    "cols": 500,
    "cmap": "turbo",
    "save_waveforms": True,
}


def _read_api_key(root: Path) -> str:
    env_key = os.environ.get("API_KEY", "").strip()
    if env_key:
        return env_key
    env_path = root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("API_KEY not found in environment or .env")


def _request_json(base_url: str, endpoint: str, method: str = "GET",
                  body: dict | None = None, api_key: str | None = None,
                  timeout_s: float = 30.0) -> dict:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["X-API-Key"] = api_key
    req = urllib.request.Request(
        base_url.rstrip("/") + endpoint,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {endpoint} failed HTTP {exc.code}: {raw}") from exc


def _wait_for_idle(base_url: str, poll_s: float = 5.0) -> dict:
    while True:
        status = _request_json(base_url, "/api/status", timeout_s=10.0)
        state = status.get("status")
        progress = status.get("progress", {})
        line = progress.get("line", 0)
        total = progress.get("total", 0)
        msg = progress.get("msg", "")
        print(f"    status={state} line={line}/{total} {msg}", flush=True)
        if state in {"COMPLETED", "STOPPED", "ERROR", "IDLE"} and not status.get("running", False):
            return status
        time.sleep(poll_s)


def _latest_meta(base_url: str) -> dict:
    meta = _request_json(base_url, "/api/data/meta", timeout_s=10.0)
    if not meta:
        raise RuntimeError("No latest C-scan metadata returned by /api/data/meta")
    return meta


def _load_scan_arrays(meta: dict) -> dict:
    session_dir = Path(meta["session_dir"])
    npz_path = session_dir / meta["feature_map_file"]
    data = np.load(npz_path)
    out = {
        "meta": meta,
        "npz_path": npz_path,
        "amplitude": np.asarray(data["amplitude"], dtype=np.float64),
        "tof": np.asarray(data["tof"], dtype=np.float64),
        "energy": np.asarray(data["energy"], dtype=np.float64),
    }
    for key in ("line_temperature_mean_c", "line_temperature_min_c",
                "line_temperature_max_c", "line_temperature_n"):
        if key in data.files:
            out[key] = np.asarray(data[key])
    return out


def _scan_temperature_summary(scan: dict) -> dict:
    mean = scan.get("line_temperature_mean_c")
    n = scan.get("line_temperature_n")
    if mean is None:
        return {"temp_mean_c": np.nan, "temp_min_c": np.nan, "temp_max_c": np.nan, "temp_n": 0}
    valid = np.isfinite(mean)
    if n is not None:
        valid &= np.asarray(n) > 0
    if not np.any(valid):
        return {"temp_mean_c": np.nan, "temp_min_c": np.nan, "temp_max_c": np.nan, "temp_n": 0}
    mins = scan.get("line_temperature_min_c", mean)
    maxs = scan.get("line_temperature_max_c", mean)
    return {
        "temp_mean_c": float(np.nanmean(mean[valid])),
        "temp_min_c": float(np.nanmin(mins[valid])),
        "temp_max_c": float(np.nanmax(maxs[valid])),
        "temp_n": int(np.nansum(n[valid])) if n is not None else int(np.sum(valid)),
    }


def _medoid_baseline(scans: list[dict], feature: str = "amplitude") -> int:
    scores = []
    arrays = [s[feature] for s in scans]
    for i, arr_i in enumerate(arrays):
        diffs = []
        for j, arr_j in enumerate(arrays):
            if i == j:
                continue
            diff = arr_j - arr_i
            diffs.append(float(np.nanmedian(np.abs(diff))))
        scores.append(float(np.nanmedian(diffs)))
    return int(np.nanargmin(scores))


def _stats(diff: np.ndarray) -> dict:
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {
            "mean": np.nan, "std": np.nan, "median_abs": np.nan,
            "p95_abs": np.nan, "min": np.nan, "max": np.nan, "range": np.nan,
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median_abs": float(np.median(np.abs(finite))),
        "p95_abs": float(np.percentile(np.abs(finite), 95)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "range": float(np.max(finite) - np.min(finite)),
    }


def _save_diff_png(diff: np.ndarray, title: str, out_path: Path, cmap: str = "coolwarm") -> None:
    finite = diff[np.isfinite(diff)]
    vmax = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    vmax = max(vmax, 1e-12)
    plt.figure(figsize=(8, 4.8), dpi=160)
    plt.imshow(diff, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(label="scan - baseline")
    plt.title(title)
    plt.xlabel("X column")
    plt.ylabel("Y line")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_noise_floor(metas: list[dict], out_dir: Path,
                        voltage_v: float | None = None) -> dict:
    scans = [_load_scan_arrays(meta) for meta in metas]
    shapes = {scan["amplitude"].shape for scan in scans}
    if len(shapes) != 1:
        raise RuntimeError(f"Scans do not have matching shapes: {sorted(shapes)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_idx = _medoid_baseline(scans, "amplitude")
    baseline = scans[baseline_idx]
    features = ("amplitude", "tof", "energy")

    rows = []
    diff_stack = {feature: [] for feature in features}
    for idx, scan in enumerate(scans):
        temp = _scan_temperature_summary(scan)
        scan_id = scan["meta"].get("scan_id", f"scan_{idx + 1}")
        base_row = {
            "scan_index": idx + 1,
            "scan_id": scan_id,
            "session_dir": scan["meta"].get("session_dir", ""),
            "is_baseline": idx == baseline_idx,
            **temp,
        }
        if idx == baseline_idx:
            for feature in features:
                rows.append({**base_row, "feature": feature, **_stats(np.zeros_like(scan[feature]))})
            continue
        for feature in features:
            diff = scan[feature] - baseline[feature]
            diff_stack[feature].append(diff)
            rows.append({**base_row, "feature": feature, **_stats(diff)})
            _save_diff_png(
                diff,
                f"{feature}: {scan_id} - {baseline['meta'].get('scan_id')}",
                out_dir / f"diff_{idx + 1:02d}_minus_baseline_{feature}.png",
            )

    span_rows = []
    npz_payload = {
        "baseline_index": np.int32(baseline_idx),
        "scan_ids": np.array([s["meta"].get("scan_id", "") for s in scans]),
    }
    for feature in features:
        if not diff_stack[feature]:
            continue
        stack = np.stack(diff_stack[feature], axis=0)
        dmin = np.nanmin(stack, axis=0)
        dmax = np.nanmax(stack, axis=0)
        span = dmax - dmin
        npz_payload[f"{feature}_diff_stack"] = stack.astype(np.float32)
        npz_payload[f"{feature}_noise_floor_min"] = dmin.astype(np.float32)
        npz_payload[f"{feature}_noise_floor_max"] = dmax.astype(np.float32)
        npz_payload[f"{feature}_noise_floor_span"] = span.astype(np.float32)
        span_stats = _stats(span)
        span_rows.append({"feature": feature, **span_stats})
        finite = span[np.isfinite(span)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
        plt.figure(figsize=(8, 4.8), dpi=160)
        plt.imshow(span, cmap="magma", vmin=0, vmax=max(vmax, 1e-12), aspect="auto")
        plt.colorbar(label="max(diff) - min(diff)")
        plt.title(f"{feature} noise-floor span across non-baseline scans")
        plt.xlabel("X column")
        plt.ylabel("Y line")
        plt.tight_layout()
        plt.savefig(out_dir / f"noise_floor_span_{feature}.png")
        plt.close()

    np.savez_compressed(out_dir / "noise_floor_diffs.npz", **npz_payload)

    csv_path = out_dir / "noise_floor_summary.csv"
    fieldnames = [
        "scan_index", "scan_id", "session_dir", "is_baseline", "feature",
        "temp_mean_c", "temp_min_c", "temp_max_c", "temp_n",
        "mean", "std", "median_abs", "p95_abs", "min", "max", "range",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    span_csv = out_dir / "noise_floor_span_summary.csv"
    with span_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean", "std", "median_abs", "p95_abs", "min", "max", "range"])
        writer.writeheader()
        writer.writerows(span_rows)

    report_path = out_dir / "NOISE_FLOOR_REPORT.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# C-scan Noise Floor Batch\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
        f.write(f"Baseline: run {baseline_idx + 1} / `{baseline['meta'].get('scan_id')}`\n\n")
        if voltage_v is not None:
            f.write(
                f"Voltage condition: {voltage_v:.3f} V at the start of the repeat batch; "
                "treated as constant unless otherwise noted.\n\n"
            )
        f.write("## Scans\n\n")
        for idx, scan in enumerate(scans):
            temp = _scan_temperature_summary(scan)
            f.write(
                f"- Run {idx + 1}: `{scan['meta'].get('scan_id')}`"
                f"{' (baseline)' if idx == baseline_idx else ''}; "
                f"temp mean={temp['temp_mean_c']:.3f} C, "
                f"range={temp['temp_min_c']:.3f}-{temp['temp_max_c']:.3f} C, "
                f"n={temp['temp_n']}\n"
            )
        f.write("\n## Noise-floor Span\n\n")
        f.write("| feature | median span | p95 span | max span |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in span_rows:
            f.write(
                f"| {row['feature']} | {row['median_abs']:.6g} | "
                f"{row['p95_abs']:.6g} | {row['max']:.6g} |\n"
            )

    return {
        "out_dir": str(out_dir),
        "baseline_index": baseline_idx + 1,
        "baseline_scan_id": baseline["meta"].get("scan_id"),
        "summary_csv": str(csv_path),
        "span_csv": str(span_csv),
        "report": str(report_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8001")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--prefix", default="noise_floor")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--voltage-v", type=float, default=None)
    ap.add_argument("--stop-temp-at-end", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    root = Path.cwd()
    api_key = _read_api_key(root)
    batch_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / "experiments" / f"noise_floor_{batch_ts}"

    metas = []
    cfg = dict(DEFAULT_CFG)
    print(f"[batch] running {args.runs} C-scans with config: {cfg}", flush=True)
    print(f"[batch] output report dir: {out_dir}", flush=True)

    try:
        for run_idx in range(1, args.runs + 1):
            scan_name = f"{args.prefix}_{batch_ts}_r{run_idx:02d}"
            body = {**cfg, "scan_name": scan_name}
            print(f"\n[batch] starting run {run_idx}/{args.runs}: {scan_name}", flush=True)
            resp = _request_json(args.base_url, "/api/start", method="POST", body=body, api_key=api_key, timeout_s=30.0)
            print(f"[batch] start response: {resp}", flush=True)
            final_status = _wait_for_idle(args.base_url, poll_s=args.poll_s)
            state = final_status.get("status")
            if state != "COMPLETED":
                raise RuntimeError(f"Run {run_idx} ended with status {state}: {final_status}")
            meta = _latest_meta(args.base_url)
            metas.append(meta)
            print(f"[batch] completed run {run_idx}: {meta.get('scan_id')} in {meta.get('session_dir')}", flush=True)
            time.sleep(2.0)

        result = analyze_noise_floor(metas, out_dir, voltage_v=args.voltage_v)
        print("\n[batch] analysis complete", flush=True)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        if args.stop_temp_at_end:
            try:
                resp = _request_json(args.base_url, "/api/temp/stop", method="POST", body={}, api_key=api_key, timeout_s=20.0)
                print(f"[batch] temp stop response: {resp}", flush=True)
            except Exception as exc:
                print(f"[batch] warning: could not stop temp logger: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
