# Data Card — OpenMarineData (Synthetic v1)

**Goal:** Provide a curated synthetic multi-sensor dataset inspired by marine monitoring systems to benchmark
streaming multivariate anomaly detection under drift.

**Sensors (example):** temperature, turbidity, oxygen, salinity, pH, chlorophyll.

**Anomaly taxonomy:**
- spike (point anomalies)
- level_shift (contextual / abrupt changes)
- collective (multi-sensor events)
- dropout (missingness / sensor failure)

**Drift regimes:**
- mean_shift
- variance_increase
- correlation_shift

**Splits:** chronological train/val/test.
