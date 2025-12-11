from __future__ import annotations
import numpy as np

def _match_with_tolerance(y_true_idx, y_pred_idx, tol: int):
    y_true_idx = list(sorted(y_true_idx))
    y_pred_idx = list(sorted(y_pred_idx))
    matched_true = set()
    matched_pred = set()

    for pi, p in enumerate(y_pred_idx):
        # find nearest unmatched true within tol
        best = None
        best_dist = None
        for ti, t in enumerate(y_true_idx):
            if ti in matched_true:
                continue
            d = abs(t - p)
            if d <= tol and (best_dist is None or d < best_dist):
                best = ti
                best_dist = d
        if best is not None:
            matched_true.add(best)
            matched_pred.add(pi)

    tp = len(matched_pred)
    fp = len(y_pred_idx) - tp
    fn = len(y_true_idx) - len(matched_true)
    return tp, fp, fn

def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    f1 = 2*p*r / (p + r + 1e-9)
    return p, r, f1

def pointwise_f1_tolerant(y_true: np.ndarray, y_pred: np.ndarray, tol: int = 5):
    true_idx = np.where(y_true.astype(int) == 1)[0]
    pred_idx = np.where(y_pred.astype(int) == 1)[0]
    tp, fp, fn = _match_with_tolerance(true_idx, pred_idx, tol)
    p, r, f1 = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

def event_f1(y_true: np.ndarray, y_pred: np.ndarray):
    # Convert contiguous anomaly regions to events
    def events(y):
        y = y.astype(int)
        ev = []
        in_ev = False
        s = 0
        for i, v in enumerate(y):
            if v == 1 and not in_ev:
                in_ev = True; s = i
            if v == 0 and in_ev:
                in_ev = False; ev.append((s, i-1))
        if in_ev:
            ev.append((s, len(y)-1))
        return ev

    true_e = events(y_true)
    pred_e = events(y_pred)

    matched = set()
    tp = 0
    for ps, pe in pred_e:
        ok = False
        for j, (ts, te) in enumerate(true_e):
            if j in matched: 
                continue
            # overlap counts
            if not (pe < ts or te < ps):
                matched.add(j); ok = True; break
        if ok: tp += 1

    fp = len(pred_e) - tp
    fn = len(true_e) - len(matched)
    p, r, f1 = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}
