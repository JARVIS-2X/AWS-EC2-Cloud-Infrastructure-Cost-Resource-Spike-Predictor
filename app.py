import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.ensemble import IsolationForest

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AWS EC2 Cost & Spike Predictor",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
    padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .4rem 0 0 0; opacity: .9;}
[data-testid="stMetric"] {
    background: rgba(125,125,125,.08);
    border: 1px solid rgba(125,125,125,.25);
    padding: 14px 18px; border-radius: 14px;
}
.banner {padding: 14px 18px; border-radius: 12px; font-weight: 600; margin-bottom: 10px;}
.banner.ok   {background:#dcfce7; color:#166534; border-left:6px solid #16a34a;}
.banner.warn {background:#fef9c3; color:#854d0e; border-left:6px solid #eab308;}
.banner.crit {background:#fee2e2; color:#991b1b; border-left:6px solid #dc2626;}
.pill {display:inline-block; padding:4px 12px; border-radius:999px; font-size:.85rem; font-weight:600;}
.pill.ok {background:#dcfce7; color:#166534;}
.pill.bad {background:#fee2e2; color:#991b1b;}
.live-dot {display:inline-block; width:10px; height:10px; border-radius:50%;
           background:#ef4444; margin-right:6px; animation:pulse 1.2s infinite;}
@keyframes pulse {0%{opacity:1} 50%{opacity:.25} 100%{opacity:1}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚡ AWS EC2 Infrastructure Cost & Resource Spike Predictor</h1>
  <p>Proactive anomaly detection, cost forecasting and actionable mitigation for EC2 instances.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# DATA + MODELS
# ----------------------------------------------------------------------------
# app.py ko CSV wale folder (archive) mein hi rakho, ya neeche path badlo
DATA_DIR = Path(__file__).resolve().parent
FALLBACK_DIR = Path(r"C:\Users\HP\Downloads\archive")
CRITICAL_CPU = 85.0
WARNING_CPU = 70.0


@st.cache_data
def load_data():
    """CSV se CPU column load karta hai. CSV na mile to synthetic data banata hai."""
    for folder in (DATA_DIR, FALLBACK_DIR):
        if folder.exists():
            for csv in sorted(folder.glob("*.csv")):
                raw = pd.read_csv(csv)
                raw.columns = raw.columns.str.strip().str.lower().str.replace(" ", "_")
                cpu_cols = [c for c in raw.columns if "cpu" in c]
                if cpu_cols:
                    s = pd.to_numeric(raw[cpu_cols[0]], errors="coerce").dropna()
                    return s.reset_index(drop=True).to_frame("cpu"), csv.name
    rng = np.random.default_rng(42)
    t = np.arange(1440)
    cpu = 40 + 28 * np.sin(2 * np.pi * t / 720) + rng.normal(0, 5, len(t))
    return pd.DataFrame({"cpu": np.clip(cpu, 0, 100)}), "synthetic data (CSV not found)"


@st.cache_resource
def train_models(values: np.ndarray):
    X = pd.DataFrame({"cpu": values})
    iso = IsolationForest(contamination=0.03, random_state=42).fit(X)

    cost_model = None
    if HAS_XGB:
        rng = np.random.default_rng(42)
        y = values * 0.05 + rng.normal(0.2, 0.05, len(values))
        cost_model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        ).fit(X, y)
    return iso, cost_model


df, source_name = load_data()
mean_cpu, std_cpu = float(df["cpu"].mean()), float(df["cpu"].std(ddof=0))
iso_forest, cost_model = train_models(df["cpu"].to_numpy())

df["z"] = (df["cpu"] - mean_cpu) / std_cpu
df["ml"] = iso_forest.predict(df[["cpu"]])  # -1 = anomaly


def assess(cpu: float) -> dict:
    """Ek CPU value ka poora analysis."""
    z = (cpu - mean_cpu) / std_cpu
    ml = int(iso_forest.predict(pd.DataFrame({"cpu": [cpu]}))[0])
    if cost_model is not None:
        hourly = float(cost_model.predict(pd.DataFrame({"cpu": [cpu]}))[0])
    else:
        hourly = cpu * 0.05 + 0.20
    if z > 3.0 or cpu > CRITICAL_CPU:
        level = "crit"
    elif z > 2.0 or cpu > WARNING_CPU:
        level = "warn"
    else:
        level = "ok"
    return {"cpu": cpu, "z": z, "ml": ml, "hourly": hourly,
            "monthly": hourly * 24 * 30, "level": level}


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def metrics_row(a: dict, prev: float | None = None):
    m1, m2, m3, m4 = st.columns(4)
    delta = None if prev is None else f"{a['cpu'] - prev:+.1f}%"
    m1.metric("CPU Usage", f"{a['cpu']:.2f}%", delta, delta_color="inverse")
    m2.metric("Hourly Cost", f"${a['hourly']:.3f}")
    m3.metric("Projected Monthly Cost", f"${a['monthly']:,.2f}")
    m4.metric("Z-Score", f"{a['z']:.2f}")


def verification_panel(a: dict):
    st.markdown("##### 🧠 Dual Verification")
    z_bad = a["z"] > 3.0
    st.markdown(
        f"**Z-Score:** <span class='pill {'bad' if z_bad else 'ok'}'>"
        f"{'Outlier Spike' if z_bad else 'Safe Boundary'}</span>",
        unsafe_allow_html=True,
    )
    ml_bad = a["ml"] == -1
    st.markdown(
        f"**Isolation Forest:** <span class='pill {'bad' if ml_bad else 'ok'}'>"
        f"{'Anomaly Detected' if ml_bad else 'Normal Pattern'}</span>",
        unsafe_allow_html=True,
    )


def action_panel(a: dict):
    st.markdown("##### 🛠️ Mitigation Plan")
    if a["level"] == "crit":
        st.markdown("<div class='banner crit'>🚨 CRITICAL: Severe resource spike detected</div>",
                    unsafe_allow_html=True)
        st.markdown(
            "- Auto-scale instance (e.g. t2.micro → t2.medium)\n"
            "- Route load via Elastic Load Balancer (ELB)\n"
            "- Throttle peak API requests to prevent a crash"
        )
    elif a["level"] == "warn":
        st.markdown("<div class='banner warn'>⚠️ WARNING: Elevated server load</div>",
                    unsafe_allow_html=True)
        st.markdown("- Monitor background processes\n- Clear server cache")
    else:
        st.markdown("<div class='banner ok'>✅ System health normal</div>",
                    unsafe_allow_html=True)
        st.caption("All parameters are within safe operational bounds.")


def gauge(cpu: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=cpu, number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#3b82f6"},
            "steps": [
                {"range": [0, WARNING_CPU], "color": "#dcfce7"},
                {"range": [WARNING_CPU, CRITICAL_CPU], "color": "#fef9c3"},
                {"range": [CRITICAL_CPU, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#dc2626", "width": 4},
                          "thickness": .8, "value": CRITICAL_CPU},
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def history_figure(data: pd.DataFrame, current: float | None = None):
    anomalies = data[data["ml"] == -1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["cpu"], mode="lines",
                             name="CPU Utilization (%)",
                             line=dict(color="#3b82f6", width=1.4)))
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["cpu"], mode="markers",
                             name="Anomalies (Isolation Forest)",
                             marker=dict(color="#ef4444", size=7,
                                         line=dict(color="white", width=1))))
    fig.add_hline(y=CRITICAL_CPU, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Critical 85%", annotation_position="top left")
    if current is not None:
        fig.add_hline(y=current, line_dash="dot", line_color="#8b5cf6",
                      annotation_text=f"Selected {current:.1f}%",
                      annotation_position="bottom right")
    fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time Index", yaxis_title="CPU Usage (%)",
        yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def live_figure(buf: list):
    d = pd.DataFrame(buf)
    bad = d[(d["level"] == "crit") | (d["ml"] == -1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["t"], y=d["cpu"], mode="lines+markers",
                             name="Live CPU (%)", line=dict(color="#3b82f6", width=2),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=bad["t"], y=bad["cpu"], mode="markers",
                             name="Anomaly / Critical",
                             marker=dict(color="#ef4444", size=11,
                                         line=dict(color="white", width=1.5))))
    fig.add_hline(y=CRITICAL_CPU, line_dash="dash", line_color="#f59e0b",
                  annotation_text="Critical 85%", annotation_position="top left")
    fig.update_layout(
        template="plotly_white", height=380, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Time", yaxis_title="CPU Usage (%)", yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        uirevision="live",
    )
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("buf", [])
ss.setdefault("idx", 0)
ss.setdefault("pending_spike", False)
ss.setdefault("cpu_val", 45.0)
ss.setdefault("running", True)

st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.radio("Select Input Mode",
                        ["Live Stream", "Manual Slider", "Random CSV Simulation"])
st.sidebar.caption(f"📁 Data source: {source_name}")

refresh = 2.0
window = 60
if mode == "Live Stream":
    st.sidebar.toggle("Stream running", key="running")
    refresh = st.sidebar.slider("Refresh interval (sec)", 0.5, 5.0, 2.0, 0.5)
    window = st.sidebar.slider("Points shown on chart", 20, 200, 60, 10)
    noise = st.sidebar.slider("Noise level", 0.0, 10.0, 2.0, 0.5)
    if st.sidebar.button("💥 Inject Spike", width="stretch"):
        ss.pending_spike = True
    if st.sidebar.button("🔄 Reset stream", width="stretch"):
        ss.buf, ss.idx = [], 0
elif mode == "Manual Slider":
    ss.cpu_val = st.sidebar.slider("CPU Utilization (%)", 0.0, 100.0,
                                   float(ss.cpu_val), 0.5)
else:
    if st.sidebar.button("🎲 Pick Random Metric From CSV", width="stretch"):
        ss.cpu_val = float(df["cpu"].sample(1).iloc[0])
    st.sidebar.info(f"Current value: **{ss.cpu_val:.2f}%**")

# ----------------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------------
tab_live, tab_hist, tab_about = st.tabs(
    ["📡 Live Monitor", "📈 Historical Analysis", "ℹ️ About"]
)

with tab_live:
    if mode == "Live Stream":
        run_every = refresh if ss.running else None

        @st.fragment(run_every=run_every)
        def live_panel():
            if ss.running or not ss.buf:
                base = df["cpu"].to_numpy()
                rng = np.random.default_rng()
                if ss.pending_spike:
                    val = rng.uniform(88, 99)
                    ss.pending_spike = False
                else:
                    val = base[ss.idx % len(base)] + rng.normal(0, noise)
                    ss.idx += 1
                val = float(np.clip(val, 0, 100))
                a = assess(val)
                a["t"] = pd.Timestamp.now()
                ss.buf.append(a)
                ss.buf = ss.buf[-500:]

            latest = ss.buf[-1]
            prev = ss.buf[-2]["cpu"] if len(ss.buf) > 1 else None

            state = "LIVE" if ss.running else "PAUSED"
            st.markdown(
                f"<span class='live-dot'></span>**{state}** · "
                f"updated {latest['t'].strftime('%H:%M:%S')}",
                unsafe_allow_html=True,
            )
            metrics_row(latest, prev)
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(live_figure(ss.buf[-window:]), width="stretch")
            with right:
                st.plotly_chart(gauge(latest["cpu"]), width="stretch")
                verification_panel(latest)
            action_panel(latest)

        live_panel()
    else:
        a = assess(float(ss.cpu_val))
        metrics_row(a)
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(history_figure(df, a["cpu"]), width="stretch")
        with right:
            st.plotly_chart(gauge(a["cpu"]), width="stretch")
            verification_panel(a)
        action_panel(a)

with tab_hist:
    n_anom = int((df["ml"] == -1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", f"{len(df):,}")
    c2.metric("Mean CPU", f"{mean_cpu:.1f}%")
    c3.metric("Std Dev", f"{std_cpu:.1f}")
    c4.metric("Anomalies (ML)", f"{n_anom} ({n_anom / len(df):.1%})")
    st.plotly_chart(history_figure(df), width="stretch")

    hist = go.Figure(go.Histogram(x=df["cpu"], nbinsx=40, marker_color="#7c3aed"))
    hist.update_layout(template="plotly_white", height=300, title="CPU Distribution",
                       xaxis_title="CPU Usage (%)", yaxis_title="Count",
                       margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(hist, width="stretch")

with tab_about:
    st.markdown(
        f"""
**How it works**

1. **Z-Score** – dataset ke real mean (`{mean_cpu:.1f}`) aur std (`{std_cpu:.1f}`) se
   har CPU value ka statistical deviation nikalta hai.
2. **Isolation Forest** – unsupervised ML model jo unusual CPU patterns ko anomaly mark karta hai.
3. **Cost forecasting** – {"XGBoost regressor" if cost_model is not None else "linear formula (XGBoost not installed)"}
   se hourly aur monthly cost predict hota hai.
4. **Mitigation engine** – Z-score aur CPU thresholds ke basis par action plan suggest karta hai.

**Thresholds:** Warning > {WARNING_CPU:.0f}% or Z > 2 · Critical > {CRITICAL_CPU:.0f}% or Z > 3
"""
    )
