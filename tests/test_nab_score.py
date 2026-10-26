import json

import pandas as pd
import pytest

from eval import nab_score as ns
from eval.data import load_windows

pytestmark = pytest.mark.skipif(not (ns.NAB_DIR / "nab" / "sweeper.py").is_file(), reason="NAB not cloned")

N = 1000
TS = pd.date_range("2020-01-01", periods=N, freq="5min")
WINDOWS = {"c/a.csv": [(TS[500], TS[549])], "c/b.csv": []}


def _runs(a_scores, b_scores=None):
    return [("c/a.csv", TS, a_scores), ("c/b.csv", TS, b_scores or [0.0] * N)]


def test_null_detector_is_zero():
    r = ns.nab_score(_runs([0.5] * N, [0.5] * N), WINDOWS)
    assert r["score"] == pytest.approx(0.0)


def test_early_detection_beats_late_and_fp_hurts():
    early = [0.0] * N; early[500] = 1.0
    late = [0.0] * N; late[545] = 1.0
    s_early = ns.nab_score(_runs(early), WINDOWS)["score"]
    s_late = ns.nab_score(_runs(late), WINDOWS)["score"]
    assert s_early > s_late > 0
    assert s_early == pytest.approx(100.0)  # earliest possible detection, no FP

    fp = [0.0] * N; fp[300] = 1.0
    with_fp = ns.nab_score(_runs(early, fp), WINDOWS)["score"]
    assert with_fp < s_early


def test_fixed_threshold():
    sc = [0.0] * N; sc[500] = 0.6
    assert ns.nab_score(_runs(sc), WINDOWS, threshold=0.5)["score"] == pytest.approx(100.0)
    # threshold above every score -> no detections -> same as null
    assert ns.nab_score(_runs(sc), WINDOWS, threshold=0.9)["score"] == pytest.approx(0.0)


@pytest.mark.parametrize("detector", ["null", "windowedGaussian", "numenta"])
def test_reproduces_published_leaderboard(detector):
    windows = load_windows(ns.NAB_DIR / "labels" / "combined_windows.json")
    published = json.loads((ns.NAB_DIR / "results" / "final_results.json").read_text())
    r = ns.nab_score(ns.runs_from_nab_results(detector, sorted(windows)), windows)
    assert r["score"] == pytest.approx(published[detector]["standard"], abs=0.01)
