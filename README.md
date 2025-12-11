# OpenMarineData (Synthetic) — Streaming Anomaly + Drift Benchmark Suite

A **versioned synthetic multi-sensor dataset** inspired by marine monitoring systems, designed to benchmark
**real-time multivariate anomaly detection** under **controlled drift regimes**.

This repository provides:
- A configurable dataset (20,000 time steps × N sensors) and a generator (anomalies + drift) 
- A clear benchmark protocol (splits, regimes, taxonomy)
- Reproducible baseline results (RoLA v2 Lite, Isolation Forest, LSTM Autoencoder)

---

## Why this exists
Real-world marine/environmental sensor streams are often hard to share due to access, licensing, or privacy.
This benchmark provides a **reproducible substitute** for:
- testing anomaly detection pipelines in streaming settings
- evaluating drift robustness (mean/variance/correlation shifts)
- comparing baselines under standard protocols

---

## Dataset summary
**Sensors (default):** temperature, turbidity, oxygen, salinity, pH, chlorophyll  
**Time split:** chronological `train/val/test`  
**Anomaly taxonomy:**
- `spike` (point anomalies)
- `level_shift` (contextual changes)
- `collective` (multi-sensor events)
- `dropout` (missingness / sensor failure)

**Drift regimes (default):**
- `mean_shift`
- `variance_increase`
- `correlation_shift`

Configuration lives in: `configs/dataset_v1.yaml`

---

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt
python src/generate_dataset.py

Outputs:
artifacts/results.json
artifacts/results.md
```

## Baselines included

- **RoLA-v2-Lite (streaming):** EWMA normalisation + correlation gate + temporal recalibration  
- **Isolation Forest (batch):** classical unsupervised baseline for anomaly scoring  
- **LSTM Autoencoder (sequence):** reconstruction-based anomaly scoring on sliding windows  

## Metrics

Results are reported in: `artifacts/results.md`

- **Pointwise F1 (tolerant window):** allows small timing offsets (±5 steps) between predicted and true anomaly points  
- **Event-level F1:** evaluates overlap between predicted anomaly segments and true anomaly segments  

See also: `docs/PROTOCOL.md`


## Reproducibility

- **Fixed seed** in `configs/dataset_v1.yaml`
- **Versioned dataset name** (e.g., `openmarinedata_synth_v1`)
- **All scripts run from the repo root**

### Suggested workflow for new experiments

1. Copy `configs/dataset_v1.yaml` → `configs/dataset_v2.yaml`
2. Modify the **drift** and/or **anomaly** regimes in the new config
3. Regenerate the dataset and rerun benchmarks:

### Project structure
configs/     # dataset versions (yaml)
data/        # generated datasets (csv + meta json)
docs/        # protocol + data card
src/
  baselines/ # reference baseline implementations
artifacts/   # benchmark outputs (results.json, results.md)

# How to Cite

If you use **OpenMarineData (Synthetic): Streaming Anomaly + Drift Benchmark Suite** in research, teaching, or derivative work, please cite the repository.

## Suggested citation (APA-style)

Orokpo, S.M.(2025). *OpenMarineData(Synthetic): Streaming Anomaly + Drift Benchmark Suite* (Version 0.1.0) [Software]. GitHub. https://github.com/freelansire/marine_dataset_benchmark
  
## BibTeX

```bibtex
@software{orokpo_openmarinedata_2025,
  author       = {Samuel Moses Orokpo},
  title        = {OpenMarineData (Synthetic): Streaming Anomaly + Drift Benchmark Suite},
  year         = {2025},
  version      = {0.1.0},
  url          = {https://github.com/freelansire/marine_dataset_benchmark},
  doi          = {10.5281/zenodo.xxxxxxx}
  note         = {Synthetic multi-sensor dataset generator and reproducible benchmark suite for streaming anomaly detection under drift regimes}
}
```
