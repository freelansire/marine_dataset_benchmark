from __future__ import annotations
import numpy as np
from collections import deque

class RoLAv2Lite:
    """
    Lightweight streaming anomaly detector:
    - per-sensor EWMA mean/std (online)
    - adaptive correlation gate using rolling correlation stability
    - temporal recalibration: dynamic threshold via recent score quantiles
    """
    def __init__(self, n_features: int, ewma_alpha=0.02, win=120, base_q=0.98):
        self.nf = n_features
        self.a = ewma_alpha
        self.win = win
        self.base_q = base_q

        self.mu = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.buf = deque(maxlen=win)
        self.score_hist = deque(maxlen=2000)

        self.prev_corr = None

    def partial_fit_score(self, x: np.ndarray) -> float:
        x = np.nan_to_num(x, nan=0.0)

        # EWMA updates
        diff = x - self.mu
        self.mu = (1 - self.a) * self.mu + self.a * x
        self.var = (1 - self.a) * self.var + self.a * (diff**2 + 1e-6)
        z = (x - self.mu) / (np.sqrt(self.var) + 1e-6)

        self.buf.append(x)
        score = float(np.max(np.abs(z)))

        # Correlation gate: if the correlation structure is stable and multiple sensors deviate together -> boost confidence
        gate = 1.0
        if len(self.buf) >= max(20, self.win // 2):
            W = np.stack(self.buf, axis=0)

            # Handle missingness and avoid zero-variance columns in correlation
            W = np.nan_to_num(W, nan=0.0)

            std = W.std(axis=0)
            valid = std > 1e-8

            if valid.sum() >= 2:
                corr_small = np.corrcoef(W[:, valid].T)

                # Expand back to full matrix
                corr = np.zeros((self.nf, self.nf), dtype=float)
                idx = np.where(valid)[0]
                for a, ia in enumerate(idx):
                    for b, ib in enumerate(idx):
                        corr[ia, ib] = corr_small[a, b]
            else:
                corr = np.zeros((self.nf, self.nf), dtype=float)

            if self.prev_corr is None:
                self.prev_corr = corr
            delta = np.mean(np.abs(corr - self.prev_corr))
            self.prev_corr = 0.9 * self.prev_corr + 0.1 * corr

            # smaller delta => structure stable => trust collective deviations more
            stability = np.clip(1.0 - 3.0 * delta, 0.2, 1.0)
            active = (np.abs(z) > 2.0).sum()
            collective = np.clip(active / max(1, self.nf), 0.0, 1.0)
            gate = 0.7 + 0.9 * stability * collective

        fused = score * gate

        # Temporal recalibration: threshold adapts with recent score distribution
        self.score_hist.append(fused)
        return fused

    def threshold(self, warm=False) -> float:
        if len(self.score_hist) < 200:
            return np.quantile(list(self.score_hist) + [1.0], 0.99)
        q = self.base_q if not warm else min(0.995, self.base_q + 0.01)
        return float(np.quantile(np.array(self.score_hist), q))
