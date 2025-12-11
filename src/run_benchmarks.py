from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils_io import load_csv, load_json, ensure_dir, save_json
from evaluate import pointwise_f1_tolerant, event_f1

from baselines.isolation_forest import IsolationForestBaseline
from baselines.lstm_ae import LSTMAEBaseline
from baselines.rola_v2_lite import RoLAv2Lite

def get_splits(df, sensors):
    def grab(split):
        d = df[df["split"] == split].copy()
        X = d[sensors].to_numpy(dtype=float)
        y = d["is_anomaly"].to_numpy(dtype=int)
        return X, y
    return grab("train"), grab("val"), grab("test")

def calibrate_threshold(scores_val: np.ndarray, expected_rate: float = 0.02):
    # threshold so top expected_rate are flagged
    q = 1.0 - expected_rate
    return float(np.quantile(scores_val, q))

def run():
    data_dir = Path("data")
    # pick the first dataset file that ends with .csv and has meta
    csvs = sorted([p for p in data_dir.glob("*.csv") if p.name.endswith(".csv")])
    if not csvs:
        raise SystemExit("No dataset found in data/. Run: python src/generate_dataset.py")

    ds_path = csvs[0]
    meta_path = data_dir / (ds_path.stem + ".meta.json")
    meta = load_json(meta_path)

    df = load_csv(ds_path)
    sensors = meta["sensors"]

    (Xtr, ytr), (Xva, yva), (Xte, yte) = get_splits(df, sensors)

    results = {"dataset": meta["name"], "baselines": {}}
    artifacts = ensure_dir("artifacts")

    # 1) Isolation Forest
    iso = IsolationForestBaseline(contamination=0.02, random_state=42)
    iso.fit(np.nan_to_num(Xtr, nan=0.0))
    s_val = iso.score(np.nan_to_num(Xva, nan=0.0))
    thr = calibrate_threshold(s_val, expected_rate=0.02)
    s_te = iso.score(np.nan_to_num(Xte, nan=0.0))
    yhat = (s_te >= thr).astype(int)

    results["baselines"]["IsolationForest"] = {
        "threshold": thr,
        "pointwise_f1_tol5": pointwise_f1_tolerant(yte, yhat, tol=5),
        "event_f1": event_f1(yte, yhat),
    }

    # 2) LSTM Autoencoder
    ae = LSTMAEBaseline(n_features=len(sensors), seq_len=40, hidden=64, epochs=5, device="cpu")
    ae.fit(Xtr)
    s_val = ae.score(Xva)
    thr = calibrate_threshold(s_val, expected_rate=0.02)
    s_te = ae.score(Xte)
    yhat = (s_te >= thr).astype(int)

    results["baselines"]["LSTM-AE"] = {
        "threshold": thr,
        "pointwise_f1_tol5": pointwise_f1_tolerant(yte, yhat, tol=5),
        "event_f1": event_f1(yte, yhat),
    }

    # 3) RoLA v2 Lite (streaming)
    rola = RoLAv2Lite(n_features=len(sensors), ewma_alpha=0.02, win=120, base_q=0.98)

    # warm-up on train stream
    for i in range(Xtr.shape[0]):
        rola.partial_fit_score(Xtr[i])

    # calibrate using val scores (still streaming)
    s_val = np.zeros(Xva.shape[0], dtype=float)
    for i in range(Xva.shape[0]):
        s_val[i] = rola.partial_fit_score(Xva[i])
    thr = calibrate_threshold(s_val, expected_rate=0.02)

    # test stream
    s_te = np.zeros(Xte.shape[0], dtype=float)
    for i in range(Xte.shape[0]):
        s_te[i] = rola.partial_fit_score(Xte[i])
    yhat = (s_te >= thr).astype(int)

    results["baselines"]["RoLA-v2-Lite"] = {
        "threshold": thr,
        "pointwise_f1_tol5": pointwise_f1_tolerant(yte, yhat, tol=5),
        "event_f1": event_f1(yte, yhat),
    }

    save_json(results, artifacts / "results.json")

    # also write a quick markdown summary
    md = ["# Benchmark Results\n", f"**Dataset:** `{meta['name']}`\n"]
    for name, res in results["baselines"].items():
        p = res["pointwise_f1_tol5"]
        e = res["event_f1"]
        md.append(f"## {name}\n")
        md.append(f"- Pointwise F1 (tol=5): **{p['f1']:.3f}** (P={p['precision']:.3f}, R={p['recall']:.3f})\n")
        md.append(f"- Event F1: **{e['f1']:.3f}** (P={e['precision']:.3f}, R={e['recall']:.3f})\n")
        md.append(f"- Threshold: `{res['threshold']:.6f}`\n")
        md.append("\n")

    (artifacts / "results.md").write_text("".join(md))
    print(f"Saved: artifacts/results.json and artifacts/results.md")

if __name__ == "__main__":
    run()
