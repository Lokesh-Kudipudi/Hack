import numpy as np
import pandas as pd

from sentinel.events import EventManager
from sentinel.pipeline import Sentinel
from sentinel.recall import ScoreSmoother, SpatialExtreme

N = 2000
TS = pd.date_range("2020-01-01", periods=N, freq="5min")


def test_spatial_extreme_flags_new_extremes_only():
    sp = SpatialExtreme(tol=0.1)
    sp.seed([0.0, 10.0])
    assert not sp.update(10.5)   # within 10 + 0.1 * 10
    assert sp.update(12.0)       # beyond -> flag, range now [0, 12]
    assert not sp.update(12.5)   # within 12 + 1.2
    assert sp.update(-3.0)


def test_score_smoother_moving_average_and_filter():
    sm = ScoreSmoother(window=3, detectors=["A"])
    assert [sm("A", v) for v in (3.0, 0.0, 0.0, 0.0)] == [3.0, 1.5, 1.0, 0.0]
    assert sm("B", 7.0) == 7.0           # not in the list -> identity
    assert ScoreSmoother(window=1)("A", 5.0) == 5.0


def test_agree_k_confirms_on_first_point():
    em = EventManager(debounce=3, cooldown=0, agree_k=2)
    assert em.update(True, top_ratio=1.1, n_fired=1) == (False, False)
    em2 = EventManager(debounce=3, cooldown=0, agree_k=2)
    assert em2.update(True, top_ratio=1.1, n_fired=2)[0] is True


def test_cooldown_frac_scales_with_probation():
    s = Sentinel(n_total=4000, config={"events": {"cooldown_frac": 0.5}})
    assert s.event_manager.cooldown_duration == int(0.5 * s.probation_limit)
    assert Sentinel(n_total=4000).event_manager.cooldown_duration == 600


def test_spatial_flag_starts_event_without_debounce():
    rng = np.random.default_rng(0)
    x = 10 + rng.normal(0, 0.5, N)
    x[1500] = 40.0  # single new extreme
    s = Sentinel(n_total=N, config={"spatial": {"enabled": True, "tol": 0.05}, "debounce": 5})
    res = [s.update(t, v) for t, v in zip(TS, x)]
    assert res[1500].is_event_start
    assert "SpatialExtreme" in res[1500].evidence["fired"]


def test_defaults_leave_recall_mechanisms_off():
    s = Sentinel(n_total=N)
    assert s.spatial is None and not s.smoother.applies("RobustZ") and s.event_manager.agree_k == 0
