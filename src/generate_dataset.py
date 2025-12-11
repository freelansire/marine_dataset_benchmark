from __future__ import annotations
import math, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from utils_io import ensure_dir, save_csv, save_json

def _set_seed(seed: int):
    np.random.seed(seed)

def _make_cov(k: int, strength: float) -> np.ndarray:
    # Simple correlated covariance: diag=1, offdiag=strength
    cov = np.full((k, k), strength, dtype=float)
    np.fill_diagonal(cov, 1.0)
    # Ensure PSD-ish
    return cov

def _ar1(n: int, phi: float, sigma: float) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    eps = np.random.normal(0, sigma, size=n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t]
    return x

def _seasonal(n: int, period: int, strength: float) -> np.ndarray:
    t = np.arange(n)
    return strength * np.sin(2 * np.pi * t / max(2, period))

def _apply_regime(arr: np.ndarray, kind: str, **kwargs) -> np.ndarray:
    out = arr.copy()
    if kind == "mean_shift":
        out += kwargs.get("magnitude", 1.0)
    elif kind == "variance_increase":
        out *= math.sqrt(kwargs.get("factor", 1.5))
    return out

def build_dataset(cfg: dict) -> tuple[pd.DataFrame, dict]:
    name = cfg["name"]
    seed = int(cfg["seed"])
    _set_seed(seed)

    sensors = cfg["sensors"]
    k = len(sensors)
    n = int(cfg["n_steps"])
    dt = int(cfg["sampling_seconds"])
    start = pd.Timestamp("2025-01-01T00:00:00Z")

    base = cfg["base"]
    period = int(base["seasonal_period_steps"])
    seasonal_strength = float(base["seasonal_strength"])
    phi = float(base["ar1"])
    noise_std = float(base["noise_std"])
    corr_strength = float(base["correlation_strength"])

    cov = _make_cov(k, corr_strength)
    L = np.linalg.cholesky(cov + 1e-6*np.eye(k))

    # Base signals: seasonal + correlated AR noise
    seasonal = _seasonal(n, period, seasonal_strength)
    X = np.zeros((n, k), dtype=float)

    # correlated noise
    Z = np.random.normal(0, 1.0, size=(n, k)) @ L.T
    for j in range(k):
        X[:, j] = seasonal + _ar1(n, phi, noise_std) + 0.4 * Z[:, j]

    # Drift regimes (annotated)
    regimes = []
    for r in cfg.get("drifts", {}).get("regimes", []):
        s = int(float(r["start_frac"]) * n)
        e = int(float(r["end_frac"]) * n)
        regimes.append({"type": r["type"], "start": s, "end": e, "params": {k:v for k,v in r.items() if k not in ["type","start_frac","end_frac"]}})

    # Apply drifts
    drift_label = np.array(["none"] * n, dtype=object)
    for reg in regimes:
        s, e = reg["start"], reg["end"]
        if reg["type"] == "correlation_shift":
            # Change covariance strength in the window by mixing independent noise
            delta = float(reg["params"].get("strength_delta", -0.2))
            mix = np.clip(0.5 + (-delta), 0.2, 0.9)  # heuristic
            X[s:e] = mix * X[s:e] + (1 - mix) * np.random.normal(0, 1.0, size=(e-s, k))
        else:
            for j in range(k):
                X[s:e, j] = _apply_regime(X[s:e, j], reg["type"], **reg["params"])
        drift_label[s:e] = reg["type"]

    # Inject anomalies with taxonomy
    an_cfg = cfg["anomalies"]
    rate = float(an_cfg["overall_rate"])
    n_anom = max(1, int(rate * n))
    types = an_cfg["types"]
    keys = list(types.keys())
    probs = np.array([types[t] for t in keys], dtype=float)
    probs /= probs.sum()

    is_anom = np.zeros(n, dtype=int)
    anom_type = np.array(["none"] * n, dtype=object)

    # choose anomaly indices mostly outside the first 5% to avoid warmup artifacts
    candidates = np.arange(int(0.05*n), n)
    idxs = np.random.choice(candidates, size=n_anom, replace=False)
    chosen_types = np.random.choice(keys, size=n_anom, p=probs)

    for idx, tname in zip(idxs, chosen_types):
        is_anom[idx] = 1
        anom_type[idx] = tname

        if tname == "spike":
            j = np.random.randint(0, k)
            X[idx, j] += np.random.normal(6.0, 1.0)
        elif tname == "level_shift":
            # short level shift window
            w = np.random.randint(30, 120)
            j = np.random.randint(0, k)
            end = min(n, idx+w)
            X[idx:end, j] += np.random.normal(2.5, 0.6)
            is_anom[idx:end] = 1
            anom_type[idx:end] = "level_shift"
        elif tname == "collective":
            # multi-sensor collective event
            w = np.random.randint(20, 80)
            end = min(n, idx+w)
            js = np.random.choice(np.arange(k), size=max(2, k//2), replace=False)
            X[idx:end][:, js] += np.random.normal(1.8, 0.5, size=(end-idx, len(js)))
            is_anom[idx:end] = 1
            anom_type[idx:end] = "collective"
        elif tname == "dropout":
            w = np.random.randint(40, 160)
            end = min(n, idx+w)
            j = np.random.randint(0, k)
            X[idx:end, j] = np.nan
            is_anom[idx:end] = 1
            anom_type[idx:end] = "dropout"

    ts = start + pd.to_timedelta(np.arange(n) * dt, unit="s")
    df = pd.DataFrame(X, columns=sensors)
    df.insert(0, "timestamp", ts.astype(str))
    df["is_anomaly"] = is_anom
    df["anomaly_type"] = anom_type
    df["drift_type"] = drift_label

    # Splits by time
    train_frac = float(cfg["splits"]["train_frac"])
    val_frac = float(cfg["splits"]["val_frac"])
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    split = np.array(["test"] * n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train+n_val] = "val"
    df["split"] = split

    meta = {
        "name": name,
        "seed": seed,
        "n_steps": n,
        "sampling_seconds": dt,
        "sensors": sensors,
        "splits": {"train": [0, n_train], "val": [n_train, n_train+n_val], "test": [n_train+n_val, n]},
        "anomaly_taxonomy": ["spike", "level_shift", "collective", "dropout"],
        "drift_regimes": regimes,
        "protocol": {
            "task": "streaming multivariate anomaly detection with drift regimes",
            "recommended_metrics": ["pointwise_f1_tolerant", "event_f1", "auprc_proxy_scores"]
        }
    }
    return df, meta

def main():
    cfg_path = Path("configs/dataset_v1.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    df, meta = build_dataset(cfg)

    ensure_dir("data")
    out_csv = Path("data") / f"{cfg['name']}.csv"
    out_meta = Path("data") / f"{cfg['name']}.meta.json"

    save_csv(df, out_csv)
    save_json(meta, out_meta)
    print(f"Saved dataset: {out_csv}")
    print(f"Saved metadata: {out_meta}")
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
