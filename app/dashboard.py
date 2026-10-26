"""Sentinel live replay dashboard.

    streamlit run app/dashboard.py

Pick a series, replay the stream point by point, and watch the value vs expected band,
ground-truth windows, flagged events, fused + per-detector scores, evidence cards and
running event metrics.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import scorer  # noqa: E402
from eval.data import data_root, list_files, load_series, load_windows, probation_length, window_indices, windows_path  # noqa: E402
from eval.run import RUNS_DIR, make_factory, sentinel_config  # noqa: E402

st.set_page_config(page_title="Sentinel — streaming anomaly detection", layout="wide")

TYPE_ICON = {"spike": "⚡", "level_shift": "📶", "variance_change": "〰️", "pattern_break": "🔁", "sequence": "🧩", "multivariate_outlier": "🌲"}


RESULTS = Path(__file__).resolve().parent.parent / "results"
# operating points: label -> settings file (None = engine defaults in sentinel/config.py)
OPERATING_POINTS = {
    "Engine default (tuned for event-F1)": None,
    "Event-F1 (results/best_config.json)": "best_config.json",
    "Joint: F1 with NAB >= 52": "best_joint_config.json",
    "NAB score (results/best_nab_config.json)": "best_nab_config.json",
    "Original K1-K8 defaults": "ORIGINAL",
}


def operating_point(label: str) -> dict:
    src = OPERATING_POINTS[label]
    if src is None:
        return {}
    if src == "ORIGINAL":
        from sentinel.config import ORIGINAL_DEFAULT_CONFIG
        return ORIGINAL_DEFAULT_CONFIG
    f = RESULTS / src
    return json.loads(f.read_text()) if f.is_file() else {}


@st.cache_data(show_spinner=False)
def corpus_index():
    root = data_root()
    files = list_files(root)
    wp = windows_path(root)
    return root, [(f.key, f.category, str(f.path)) for f in files], load_windows(wp) if wp else {}


def _expected_and_sigma(det):
    """Duck-typed: works with any Sentinel exposing .detectors with expected()/sigma()."""
    dets = getattr(det, "detectors", None) or []
    if isinstance(dets, dict):
        dets = list(dets.values())
    by = {getattr(d, "name", ""): d for d in dets}
    exp, sig = None, None
    for name in ("Seasonal", "LaggedResidual", "RobustZ"):
        d = by.get(name)
        if d is not None and hasattr(d, "expected") and d.expected() is not None:
            exp = d.expected()
            sig = _sigma(d)
            break
    if sig is None:
        sig = _sigma(by.get("RobustZ"))
    return exp, sig


def _sigma(d):
    if d is None:
        return None
    if hasattr(d, "sigma"):
        return d.sigma()
    return getattr(d, "scale", None)


@st.cache_data(show_spinner="Streaming the series through the detector…")
def replay(path: str, key: str, detector: str, config_json: str) -> tuple:
    df = load_series(Path(path))
    factory = make_factory(detector, json.loads(config_json) if detector == "sentinel" else {})
    det = factory(len(df), key)
    rows, exp, sig, dscores = [], [], [], []
    t0 = time.perf_counter()
    for t, x in zip(df["timestamp"].tolist(), df["value"].tolist()):
        r = det.update(t, x)
        rows.append(r)
        e, s = _expected_and_sigma(det)
        exp.append(e)
        sig.append(s)
        dscores.append(getattr(r, "detector_scores", {}) or {})
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame({
        "timestamp": df["timestamp"], "value": df["value"],
        "score": [r.score for r in rows], "label": [r.label for r in rows],
        "is_event_start": [bool(r.is_event_start) for r in rows], "evidence": [r.evidence for r in rows],
        "expected": pd.to_numeric(pd.Series(exp), errors="coerce"),
        "sigma": pd.to_numeric(pd.Series(sig), errors="coerce"),
    })
    lanes = pd.DataFrame(dscores).astype(float)
    return out, lanes, 1e6 * elapsed / len(df)


def surprise(score) -> np.ndarray:
    """-log10 of the tail probability (score = 1 - p): 4 means a 1-in-10,000 event."""
    return -np.log10(np.clip(1.0 - np.asarray(score, dtype=float), 1e-12, 1.0))


def make_figure(out: pd.DataFrame, lanes: pd.DataFrame, upto: int, wins_idx, probation: int, band_k: float,
                alarm_line: float = None):
    o = out.iloc[:upto]
    has_lanes = not lanes.empty
    fig = make_subplots(rows=3 if has_lanes else 2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.6, 0.15, 0.25] if has_lanes else [0.8, 0.2])
    ts = out["timestamp"]
    if o["expected"].notna().any():
        e, s = o["expected"], o["sigma"].fillna(0) * band_k
        fig.add_trace(go.Scatter(x=o["timestamp"], y=e + s, line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=o["timestamp"], y=e - s, fill="tonexty", fillcolor="rgba(26,115,232,0.12)",
                                 line=dict(width=0), name=f"expected ±{band_k:g}σ", hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=o["timestamp"], y=e, line=dict(color="#1a73e8", width=1, dash="dot"),
                                 name="expected"), 1, 1)
    fig.add_trace(go.Scatter(x=o["timestamp"], y=o["value"], line=dict(color="#202124", width=1), name="value"), 1, 1)
    an = o[o["label"] == "ANOMALOUS"]
    fig.add_trace(go.Scatter(x=an["timestamp"], y=an["value"], mode="markers", marker=dict(color="#fbbc04", size=4),
                             name="anomalous point"), 1, 1)
    ev = o[o["is_event_start"]]
    fig.add_trace(go.Scatter(x=ev["timestamp"], y=ev["value"], mode="markers",
                             marker=dict(color="#d93025", size=11, symbol="triangle-down", line=dict(color="white", width=1)),
                             name="event start",
                             text=[(e or {}).get("text", "") for e in ev["evidence"]], hoverinfo="text+x"), 1, 1)
    fig.add_trace(go.Scatter(x=o["timestamp"], y=surprise(o["score"]), line=dict(color="#d93025", width=1),
                             name="fused surprise", showlegend=False, customdata=o["score"],
                             hovertemplate="score %{customdata:.6f}<extra></extra>"), 2, 1)
    if alarm_line:
        fig.add_hline(y=alarm_line, line=dict(color="#d93025", dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(title_text="−log₁₀ p", row=2, col=1)
    if has_lanes:
        lz = lanes.iloc[:upto]
        fig.add_trace(go.Heatmap(x=o["timestamp"], y=list(lz.columns), z=surprise(lz.T.values), colorscale="Reds",
                                 zmin=1, zmax=alarm_line or 6, showscale=False,
                                 hovertemplate="%{y}: −log₁₀ p = %{z:.1f}<extra></extra>"), 3, 1)
        fig.update_yaxes(title_text="detectors", row=3, col=1)
    # probation + ground-truth windows (added after traces: plotly skips empty subplots)
    fig.add_vrect(x0=ts.iloc[0], x1=ts.iloc[min(probation, len(ts) - 1)], fillcolor="#9aa0a6", opacity=0.15,
                  line_width=0, row="all", col=1, exclude_empty_subplots=False)
    for i, j in wins_idx:
        fig.add_vrect(x0=ts.iloc[i], x1=ts.iloc[j], fillcolor="#34a853", opacity=0.18, line_width=0, row="all", col=1,
                      exclude_empty_subplots=False, layer="below")
    for name, color in (("GT anomaly window", "rgba(52,168,83,0.35)"), ("probation", "rgba(154,160,166,0.35)")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol="square", size=12, color=color)), 1, 1)
    fig.update_xaxes(range=[ts.iloc[0], ts.iloc[-1]])
    fig.update_layout(height=720, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.04),
                      hovermode="x unified" if upto < 6000 else "closest", plot_bgcolor="white")
    return fig


def live_metrics(out, windows, upto, probation):
    ts = out["timestamp"].iloc[:upto]
    t_now = ts.iloc[-1]
    started = [(a, b) for a, b in windows if a <= t_now]
    fs = scorer.score_file(ts, out["is_event_start"].iloc[:upto], started, probation=probation)
    pending = sum(1 for a, b in started if b > t_now)
    return fs, pending


def main():
    root, files, windows = corpus_index()
    st.sidebar.title("🛰️ Sentinel")
    st.sidebar.caption(f"data: `{root}`")
    cats = sorted({c for _, c, _ in files})
    default_cat = cats.index("realKnownCause") if "realKnownCause" in cats else 0
    cat = st.sidebar.selectbox("Category", cats, index=default_cat)
    keys = [(k, p) for k, c, p in files if c == cat]
    names = [k.split("/", 1)[1] for k, _ in keys]
    default_file = names.index("nyc_taxi.csv") if "nyc_taxi.csv" in names else 0
    name = st.sidebar.selectbox("Series", names, index=default_file)
    key, path = keys[names.index(name)]

    detector = st.sidebar.radio("Detector", ["sentinel", "windowed_gaussian"], horizontal=True)
    op = st.sidebar.selectbox("Operating point", list(OPERATING_POINTS), disabled=detector != "sentinel",
                              help="Event-F1 and NAB favour different settings (results/figures/pareto_f1_nab.png)")
    cfg_text = st.sidebar.text_area("Sentinel config overrides (JSON)", json.dumps(operating_point(op), indent=1),
                                    height=150, disabled=detector != "sentinel", key=f"cfg_{op}")
    try:
        cfg = json.loads(cfg_text or "{}")
    except json.JSONDecodeError as e:
        st.sidebar.error(f"bad JSON: {e}")
        cfg = {}
    band_k = st.sidebar.slider("Band width (σ)", 1.0, 6.0, 3.0, 0.5)

    try:
        out, lanes, us = replay(path, key, detector, json.dumps(cfg, sort_keys=True))
    except ImportError:
        st.error("`sentinel` package not found — run with the baseline detector or pull Likhit's engine.")
        return
    n = len(out)
    probation = probation_length(n)
    wins = windows.get(key, [])
    wins_idx = window_indices(out["timestamp"], wins)

    st.title(f"{key}")
    if "upto" not in st.session_state or st.session_state.get("key") != key:
        st.session_state.upto, st.session_state.key = n, key
    c1, c2, c3 = st.columns([6, 1, 1])
    step = max(n // 60, 50)
    with c2:
        play = st.button("▶ Replay", width="stretch")
    with c3:
        if st.button("⏭ End", width="stretch"):
            st.session_state.upto = n
    with c1:
        upto = st.slider("Stream position", min_value=min(probation + 1, n), max_value=n, value=st.session_state.upto,
                         step=1, label_visibility="collapsed")
    st.session_state.upto = upto

    metrics_ph, chart_ph = st.empty(), st.empty()

    def render(u):
        fs, pending = live_metrics(out, wins, u, probation)
        m = scorer.metrics([fs])
        with metrics_ph.container():
            k = st.columns(7)
            k[0].metric("Points streamed", f"{100 * u / n:.0f}%", help=f"{u:,} of {n:,}")
            k[1].metric("Events flagged", int(out["is_event_start"].iloc[probation:u].sum()))
            k[2].metric("TP windows", fs.tp)
            k[3].metric("FP events", fs.fp)
            k[4].metric("Missed windows", fs.fn, help=f"{pending} window(s) still open")
            k[5].metric("Event-F1 so far", "—" if m["f1"] is None else f"{m['f1']:.2f}")
            k[6].metric("µs / point", f"{us:.1f}")
        alarm = -np.log10(sentinel_config(cfg)["spot"]["q"]) if detector == "sentinel" else None
        chart_ph.plotly_chart(make_figure(out, lanes, u, wins_idx, probation, band_k, alarm), width="stretch")

    if play:
        for u in range(min(probation + step, n), n + 1, step):
            render(u)
            time.sleep(0.05)
        upto = st.session_state.upto = n
    render(upto)

    st.subheader("Evidence")
    ev = out.iloc[:upto]
    ev = ev[ev["is_event_start"]]
    if ev.empty:
        st.info("No events flagged yet.")
    for idx, r in ev.iloc[::-1].iterrows():
        e = r["evidence"] or {}
        in_win = any(i <= idx <= j for i, j in wins_idx)
        badge = "✅ inside an anomaly window" if in_win else "❌ outside all anomaly windows"
        with st.expander(f"{TYPE_ICON.get(e.get('type'), '•')} {r['timestamp']} — {e.get('type', 'anomaly')} · "
                         f"score {r['score']:.6f} · {badge}", expanded=False):
            st.write(e.get("text", ""))
            st.json({k: v for k, v in e.items() if k != "text"})


main()
