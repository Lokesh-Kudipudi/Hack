"""Multi-Stream Pipeline Pool for concurrent streaming anomaly detection (Task K17).

Provides lazy initialization, lifecycle management, per-stream routing,
and memory accounting across thousands of independent time-series streams.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from sentinel.pipeline import Sentinel
from sentinel.types import Result


class SentinelPool:
    """Manages an active collection of independent Sentinel pipelines by stream ID."""

    def __init__(
        self,
        default_config: Optional[Dict[str, Any]] = None,
        default_probation: int = 300,
        max_streams: Optional[int] = None,
    ):
        self.default_config = default_config
        self.default_probation = default_probation
        self.max_streams = max_streams
        self._pool: Dict[str, Sentinel] = {}
        self._meta: Dict[str, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def stream_ids(self) -> List[str]:
        """List all active stream identifiers."""
        return list(self._pool.keys())

    def get_or_create(self, stream_id: str, config: Optional[Dict[str, Any]] = None) -> Sentinel:
        """Get existing stream Sentinel or lazily instantiate a new one."""
        if stream_id in self._pool:
            return self._pool[stream_id]

        if self.max_streams is not None and len(self._pool) >= self.max_streams:
            raise RuntimeError(
                f"SentinelPool capacity reached ({self.max_streams} streams). "
                "Evict or close inactive streams before adding new ones."
            )

        active_cfg = config or self.default_config
        sentinel = Sentinel(n_total=None, config=active_cfg)
        self._pool[stream_id] = sentinel
        self._meta[stream_id] = {
            "created_at": pd.Timestamp.now(),
            "total_points": 0,
            "total_anomalies": 0,
            "last_active": None,
        }
        return sentinel

    def update(
        self,
        stream_id: str,
        t: Any,
        x: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Result:
        """Route an observation (t, x) to its corresponding stream detector."""
        sentinel = self.get_or_create(stream_id, config=config)
        res = sentinel.update(t, x)

        meta = self._meta[stream_id]
        meta["total_points"] += 1
        meta["last_active"] = res.timestamp
        if res.label == "ANOMALOUS":
            meta["total_anomalies"] += 1

        return res

    def close_stream(self, stream_id: str) -> bool:
        """Close and release memory for a retired stream."""
        if stream_id in self._pool:
            del self._pool[stream_id]
            del self._meta[stream_id]
            return True
        return False

    def memory_report(self) -> Dict[str, Any]:
        """Generate a memory accounting report across all managed streams."""
        total_points = sum(m["total_points"] for m in self._meta.values())
        total_anomalies = sum(m["total_anomalies"] for m in self._meta.values())

        # Estimate buffer memory usage
        estimated_bytes = sys.getsizeof(self._pool) + sys.getsizeof(self._meta)
        per_stream_stats = {}

        for sid, sentinel in self._pool.items():
            s_bytes = sys.getsizeof(sentinel)
            # Count deque and dict sizes
            for d in sentinel.detectors.values():
                s_bytes += sys.getsizeof(d)
                for attr in dir(d):
                    val = getattr(d, attr, None)
                    if hasattr(val, "__len__"):
                        s_bytes += sys.getsizeof(val)

            estimated_bytes += s_bytes
            per_stream_stats[sid] = {
                "points": self._meta[sid]["total_points"],
                "anomalies": self._meta[sid]["total_anomalies"],
                "in_probation": sentinel.count <= sentinel.probation_limit,
                "in_event": sentinel.in_event,
                "est_kb": round(s_bytes / 1024, 2),
            }

        return {
            "active_streams": len(self._pool),
            "total_points_processed": total_points,
            "total_anomalies_flagged": total_anomalies,
            "estimated_total_memory_kb": round(estimated_bytes / 1024, 2),
            "streams": per_stream_stats,
        }

    def save_pool(self, dir_path: Union[str, Path]) -> None:
        """Save checkpoints for all streams in the pool to a directory."""
        target_dir = Path(dir_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        for sid, sentinel in self._pool.items():
            clean_name = sid.replace("/", "_").replace("\\", "_")
            sentinel.save(target_dir / f"{clean_name}.json")

    @classmethod
    def load_pool(cls, dir_path: Union[str, Path]) -> SentinelPool:
        """Restore a pool of streams from a directory of checkpoints."""
        source_dir = Path(dir_path)
        pool = cls()
        if not source_dir.exists():
            return pool

        for p in source_dir.glob("*.json"):
            stream_id = p.stem
            sentinel = Sentinel.load(p)
            pool._pool[stream_id] = sentinel
            pool._meta[stream_id] = {
                "created_at": pd.Timestamp.now(),
                "total_points": sentinel.count,
                "total_anomalies": 0,
                "last_active": sentinel._last_t,
            }

        return pool
