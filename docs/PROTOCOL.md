# Benchmark Protocol

## Task
Streaming multivariate anomaly detection under drift.

## Inputs
Time-indexed sensor streams with missingness.

## Outputs
Per-timestep anomaly score + binary label.

## Recommended metrics
- Pointwise F1 with tolerance window (±5 steps)
- Event-level F1 (overlap-based)

## Baselines included
- Isolation Forest (batch)
- LSTM Autoencoder (sequence reconstruction)
- RoLA-v2-Lite (streaming, adaptive threshold + correlation gate)
