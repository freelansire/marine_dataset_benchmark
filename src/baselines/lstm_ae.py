from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.LSTM(n_features, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, n_features, batch_first=True)

    def forward(self, x):
        z, _ = self.enc(x)
        y, _ = self.dec(z)
        return y

class LSTMAEBaseline:
    def __init__(self, n_features: int, seq_len: int = 40, hidden: int = 64, lr: float = 1e-3, epochs: int = 5, device: str = "cpu"):
        self.seq_len = seq_len
        self.device = device
        self.model = LSTMAE(n_features, hidden).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = epochs

    def _make_seqs(self, X):
        # X: [T, F] -> sequences [N, L, F]
        T, F = X.shape
        if T <= self.seq_len:
            return np.empty((0, self.seq_len, F))
        seqs = np.stack([X[i:i+self.seq_len] for i in range(T - self.seq_len)], axis=0)
        return seqs

    def fit(self, X: np.ndarray):
        X = np.nan_to_num(X, nan=0.0)
        seqs = self._make_seqs(X)
        if len(seqs) == 0:
            return
        data = torch.tensor(seqs, dtype=torch.float32, device=self.device)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(data.size(0))
            for i in range(0, data.size(0), 64):
                batch = data[perm[i:i+64]]
                pred = self.model(batch)
                loss = ((pred - batch) ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0)
        seqs = self._make_seqs(X)
        T = X.shape[0]
        scores = np.zeros(T, dtype=float)

        if len(seqs) == 0:
            return scores

        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(seqs, dtype=torch.float32, device=self.device)
            pred = self.model(data)
            err = ((pred - data) ** 2).mean(dim=(1,2)).detach().cpu().numpy()

        # align: assign error to sequence end index
        for i, e in enumerate(err):
            scores[i + self.seq_len] = e
        return scores
