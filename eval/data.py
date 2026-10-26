"""Corpus loading: NAB-style folders of `timestamp,value` CSVs + a windows JSON.

Layout-agnostic: CSVs are found recursively and their category is the name of
the directory that contains them, so all of these work:
    data/NAB/data/<category>/<file>.csv          (git clone of numenta/NAB)
    data/<category>/<category>/<file>.csv        (starter pack zip extraction)
    <any folder>/<category>/<file>.csv
The file key is "<category>/<file>.csv", matching NAB's combined_windows.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_FILENAME = "combined_windows.json"
PROBATION_FRACTION = 0.15
PROBATION_MAX = 750

Window = Tuple[pd.Timestamp, pd.Timestamp]


@dataclass(frozen=True)
class FileInfo:
    key: str        # "<category>/<name>.csv"
    category: str
    name: str
    path: Path


def data_root() -> Path:
    """$SENTINEL_DATA, else data/NAB if cloned, else data/."""
    env = os.environ.get("SENTINEL_DATA")
    if env:
        return Path(env).expanduser().resolve()
    nab = REPO_ROOT / "data" / "NAB"
    return nab if nab.is_dir() else REPO_ROOT / "data"


def _csv_dir(root: Path) -> Path:
    # A NAB clone also has CSVs under results/; only read its data/ folder.
    return root / "data" if (root / "data").is_dir() and (root / "labels").is_dir() else root


def windows_path(root: Optional[Path] = None) -> Optional[Path]:
    """$SENTINEL_WINDOWS, else root/labels/, else first match under root, else data/NAB/labels/."""
    env = os.environ.get("SENTINEL_WINDOWS")
    if env:
        return Path(env).expanduser().resolve()
    root = Path(root) if root else data_root()
    candidates = [root / "labels" / WINDOWS_FILENAME]
    candidates += sorted(p for p in root.rglob(WINDOWS_FILENAME) if not _skip(p.relative_to(root)))
    candidates.append(REPO_ROOT / "data" / "NAB" / "labels" / WINDOWS_FILENAME)
    return next((p for p in candidates if p.is_file()), None)


def _skip(p: Path) -> bool:
    return any(part.startswith((".", "__MACOSX")) for part in p.parts)


def list_files(root: Optional[Path] = None, categories: Optional[Sequence[str]] = None) -> List[FileInfo]:
    root = Path(root) if root else data_root()
    base = _csv_dir(root)
    out = []
    for p in sorted(base.rglob("*.csv")):
        rel = p.relative_to(base)
        # skip hidden/__MACOSX, NAB detector outputs, and a NAB clone nested under a starter-pack root
        if _skip(rel) or "results" in rel.parts or (len(rel.parts) > 1 and (base / rel.parts[0] / "labels").is_dir()):
            continue
        cat = p.parent.name
        if categories and cat not in categories:
            continue
        out.append(FileInfo(key=f"{cat}/{p.name}", category=cat, name=p.name, path=p))
    return out


def load_series(path: Path) -> pd.DataFrame:
    """DataFrame with columns timestamp (datetime64), value (float), in file (= stream) order.

    Not sorted: a few NAB files have duplicate or backward timestamps and NAB streams them as-is.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if not {"timestamp", "value"} <= set(df.columns):
        raise ValueError(f"{path}: expected columns timestamp,value; got {list(df.columns)}")
    df = df[["timestamp", "value"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = df["value"].astype(float)
    return df.reset_index(drop=True)


def load_windows(path: Optional[Path] = None) -> Dict[str, List[Window]]:
    path = Path(path) if path else windows_path()
    if path is None:
        raise FileNotFoundError(f"{WINDOWS_FILENAME} not found; set SENTINEL_WINDOWS")
    with open(path) as f:
        raw = json.load(f)
    return {k: [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in v] for k, v in raw.items()}


def probation_length(n: int) -> int:
    return min(int(PROBATION_FRACTION * n), PROBATION_MAX)


def window_indices(timestamps: pd.Series, windows: Sequence[Window]) -> List[Tuple[int, int]]:
    """Map timestamp windows to inclusive index ranges [i_start, i_end]; drops windows with no points.

    i = first point with t >= start, j = last point with t <= end (robust to non-monotonic time).
    """
    ts = pd.DatetimeIndex(timestamps).values
    out = []
    for a, b in windows:
        ge = np.flatnonzero(ts >= np.datetime64(a))
        le = np.flatnonzero(ts <= np.datetime64(b))
        if len(ge) and len(le) and ge[0] <= le[-1]:
            out.append((int(ge[0]), int(le[-1])))
    return out


def iter_corpus(
    root: Optional[Path] = None,
    categories: Optional[Sequence[str]] = None,
    windows: Optional[Dict[str, List[Window]]] = None,
) -> Iterator[Tuple[FileInfo, pd.DataFrame, List[Window]]]:
    """Yield (info, series, windows) per file. Files missing from the windows JSON get []."""
    root = Path(root) if root else data_root()
    if windows is None:
        wp = windows_path(root)
        windows = load_windows(wp) if wp else {}
    for info in list_files(root, categories):
        yield info, load_series(info.path), windows.get(info.key, [])
