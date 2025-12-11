from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))

def load_json(path: str | Path):
    return json.loads(Path(path).read_text())

def save_csv(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
