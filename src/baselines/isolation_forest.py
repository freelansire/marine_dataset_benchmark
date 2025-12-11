from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest

class IsolationForestBaseline:
    def __init__(self, contamination=0.02, random_state=42):
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        # Higher score => more anomalous
        s = -self.model.score_samples(X)
        return s
