from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from api_client import predict_degradation_curve
from data_utils import (
    TIER_COLOUR,
    get_demo_engines,
    parse_uploaded_csv,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RUL Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* Dividers */
hr { border-color: #21262d; margin: 1.5rem 0; }

/* Metric overrides */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label { color: #8b949e; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; }

/* Cards */
.engine-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1.2rem 1.4rem 0.8rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s;
}
.engine-card:hover { border-color: #30363d; }

/* Status badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'IBM Plex Mono', monospace;
}

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
}

/* Section headers */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #21262d;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: #f59e0b;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #161b22;
    border-color: #30363d;
    color: #e6edf3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def risk_tier(rul: float) -> str:
    if rul < 30:   return "CRITICAL"
    if rul <= 80:  return "WARNING"
    return "HEALTHY"


@st.cache_data(show_spinner=False)
def cached_curve(subset, engine_id, _data_dir, _base_url, _endpoint):
    from data_utils import load_test_engine
    engine_df = load_test_engine(subset, engine_id, data_dir=_data_dir)
    return predict_degradation_curve(engine_df, subset, base_url=_base_url, endpoint=_endpoint)


def gauge_chart(predicted: float, true_rul: float | None, tier: str) -> go.Figure:
    colour = TIER_COLOUR[tier]
    max_val = 130

    fig = go.Figure()

    # Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predicted,
        number=dict(
            font=dict(family="IBM Plex Mono", size=36, color=colour),
            suffix=" cycles",
        ),
        gauge=dict(
            axis=dict(
                range=[0, max_val],
                tickfont=dict(family="IBM Plex Mono", size=10, color="#8b949e"),
                tickcolor="#21262d",
            ),
            bar=dict(color=colour, thickness=0.25),
            bgcolor="#0d1117",
            borderwidth=0,
            steps=[
                dict(range=[0, 30],  color="rgba(239,68,68,0.12)"),
                dict(range=[30, 80], color="rgba(245,158,11,0.08)"),
                dict(range=[80, max_val], color="rgba(16,185,129,0.08)"),
            ],
            threshold=dict(
                line=dict(color="#8b949e", width=2),
                thickness=0.75,
                value=true_rul if true_rul is not None else predicted,
            ),
        ),
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"),
    )
    return fig


def predicted_vs_actual_chart(predicted: float, true_rul: float, tier: str) -> go.Figure:
    colour = TIER_COLOUR[tier]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Actual", "Predicted"],
        y=[true_rul, predicted],
        marker_color=["#30363d", colour],
        marker_line_width=0,
        text=[f"{true_rul:.0f}", f"{predicted:.0f}"],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=13, color=["#8b949e", colour]),
        width=0.45,
    ))

    error = predicted - true_rul
    error_str = f"{'↑' if error >= 0 else '↓'} {abs(error):.1f} cycles"
    error_colour = "#ef4444" if abs(error) > 20 else "#10b981"

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, max(true_rul, predicted) * 1.35],
            showgrid=True,
            gridcolor="#21262d",
            tickfont=dict(family="IBM Plex Mono", size=10, color="#8b949e"),
            zeroline=False,
        ),
        xaxis=dict(
            tickfont=dict(family="IBM Plex Sans", size=12, color="#8b949e"),
            showgrid=False,
        ),
        showlegend=False,
        title=dict(
            text=f"<span style='font-size:11px;color:#8b949e;'>Error: </span>"
                 f"<span style='font-size:11px;color:{error_colour};font-family:IBM Plex Mono;'>{error_str}</span>",
            x=0.5, xanchor="center",
            font=dict(size=11),
        ),
        bargap=0.4,
    )
    return fig


def degradation_chart(cycles, ruls, tier: str) -> go.Figure:
    colour = TIER_COLOUR[tier]

    fig = go.Figure()

    # Shaded risk zones
    max_cycle = max(cycles)
    for ymin, ymax, zone_colour in [
        (0, 30, "rgba(239,68,68,0.06)"),
        (30, 80, "rgba(245,158,11,0.06)"),
        (80, 130, "rgba(16,185,129,0.06)"),
    ]:
        fig.add_hrect(y0=ymin, y1=ymax, fillcolor=zone_colour, line_width=0)

    # Threshold lines
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(239,68,68,0.4)", line_width=1)
    fig.add_hline(y=80, line_dash="dot",  line_color="rgba(245,158,11,0.4)", line_width=1)

    # Degradation curve
    fig.add_trace(go.Scatter(
        x=cycles,
        y=ruls,
        mode="lines",
        line=dict(color=colour, width=2, shape="spline", smoothing=0.4),
        fill="tozeroy",
        fillcolor={"CRITICAL":"rgba(239,68,68,0.09)","WARNING":"rgba(245,158,11,0.09)","HEALTHY":"rgba(16,185,129,0.09)"}[tier],
        hovertemplate="Cycle %{x}<br>Predicted RUL: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=30, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Cycle", font=dict(size=10, color="#8b949e")),
            tickfont=dict(family="IBM Plex Mono", size=9, color="#8b949e"),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 130],
            title=dict(text="RUL", font=dict(size=10, color="#8b949e")),
            tickfont=dict(family="IBM Plex Mono", size=9, color="#8b949e"),
            showgrid=True,
            gridcolor="#21262d",
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


def render_engine_card(col, label, tier, true_rul, cycles, ruls):
    with col:
        colour   = TIER_COLOUR[tier]
        last_rul = ruls[-1]

        st.markdown(f"""
        <div class="engine-card" style="border-top: 3px solid {colour};">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                <span style="font-weight:700; font-size:15px;">{label}</span>
                <span class="badge" style="background:{colour}33; color:{colour};">{tier}</span>
            </div>
            <div style="display:flex; gap:2rem; margin-bottom:0.5rem;">
                <div>
                    <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Predicted RUL</div>
                    <div style="font-family:'IBM Plex Mono',monospace; font-size:2rem; font-weight:600; color:{colour}; line-height:1.1;">{last_rul:.0f}<span style="font-size:13px; color:#8b949e; font-weight:400;"> cycles</span></div>
                </div>
                {f'''<div>
                    <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Actual RUL</div>
                    <div style="font-family:IBM Plex Mono,monospace; font-size:2rem; font-weight:600; color:#e6edf3; line-height:1.1;">{true_rul}<span style="font-size:13px; color:#8b949e; font-weight:400;"> cycles</span></div>
                </div>
                <div>
                    <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em;">Error</div>
                    <div style="font-family:IBM Plex Mono,monospace; font-size:2rem; font-weight:600; color:{"#ef4444" if abs(last_rul - true_rul) > 20 else "#10b981"}; line-height:1.1;">{last_rul - true_rul:+.0f}<span style="font-size:13px; color:#8b949e; font-weight:400;"> cycles</span></div>
                </div>''' if true_rul is not None else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Degradation Curve", "Predicted vs Actual"])
        with tab1:
            st.plotly_chart(degradation_chart(cycles, ruls, tier), use_container_width=True)
        with tab2:
            if true_rul is not None:
                st.plotly_chart(predicted_vs_actual_chart(last_rul, true_rul, tier), use_container_width=True)
            else:
                st.info("Upload a file with known RUL to see actual vs predicted comparison.")


# ── sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    base_url = st.text_input("API base URL", value="http://localhost:8000")
    data_dir = st.text_input("Data directory", value="data/raw")
    st.divider()
    st.markdown("""
    **Risk tiers**
    🔴 CRITICAL — < 30 cycles
    🟠 WARNING — 30–80 cycles
    🟢 HEALTHY — > 80 cycles
    """)


# ── header ────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([3, 1], gap="medium")

with hdr_left:
    st.markdown("""
    <div style="display:flex; align-items:baseline; gap:1rem; margin-bottom:0.25rem;">
        <h1 style="font-size:1.6rem; font-weight:700; margin:0;">✈️ RUL Predictor</h1>
        <span style="color:#8b949e; font-size:13px;">NASA CMAPSS · multi-condition</span>
    </div>
    """, unsafe_allow_html=True)

with hdr_right:
    st.markdown("<div style='padding-top:0.4rem;'></div>", unsafe_allow_html=True)
    use_lstm = st.toggle("Use LSTM model", value=False)
    model_endpoint = "/predict/lstm" if use_lstm else "/predict"
    model_label = "LSTM  —  FD001 RMSE 11.79" if use_lstm else "XGBoost  —  FD001 RMSE 13.86"
    colour = "#10b981" if use_lstm else "#f59e0b"
    st.markdown(
        f"<span style='font-family:IBM Plex Mono,monospace; font-size:11px; color:{colour};'>{model_label}</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ── demo engines ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Demo Engines — FD001 Test Set</div>', unsafe_allow_html=True)

try:
    demo_specs = get_demo_engines(data_dir=data_dir)
except FileNotFoundError:
    st.error(f"Test data not found in `{data_dir}`. Update the **Data directory** in the sidebar (☰).")
    st.stop()

demo_cols   = st.columns(3, gap="medium")
demo_results = []

for spec in demo_specs:
    with st.spinner(f"Computing {spec['label']} …"):
        try:
            cycles, ruls = cached_curve(
                spec["subset"], spec["engine_id"], data_dir, base_url, model_endpoint
            )
            demo_results.append((spec, cycles, ruls))
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

for (spec, cycles, ruls), col in zip(demo_results, demo_cols):
    render_engine_card(
        col=col,
        label=spec["label"],
        tier=spec["tier"],
        true_rul=spec["true_rul"],
        cycles=cycles,
        ruls=ruls,
    )

# ── upload ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-label">Custom Engine Upload</div>', unsafe_allow_html=True)

up_col, cfg_col = st.columns([3, 1], gap="medium")
with up_col:
    uploaded_file = st.file_uploader(
        "Raw CMAPSS format · space-separated · no header · 26 columns · ≥ 30 cycles",
        type=["txt", "csv"],
        label_visibility="visible",
    )
with cfg_col:
    upload_subset = st.selectbox("Subset", ["FD001", "FD002", "FD003", "FD004"])

if uploaded_file is not None:
    try:
        engine_df = parse_uploaded_csv(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    n_windows = len(engine_df) - 30 + 1
    progress_bar = st.progress(0, text=f"Running {n_windows} predictions …")

    def _on_progress(current, total):
        progress_bar.progress(current / total, text=f"Predicting window {current}/{total} …")

    try:
        up_cycles, up_ruls = predict_degradation_curve(
            engine_df=engine_df,
            subset=upload_subset,
            base_url=base_url,
            endpoint=model_endpoint,
            on_progress=_on_progress,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    progress_bar.empty()
    tier = risk_tier(up_ruls[-1])

    up_card_col, _, _ = st.columns(3, gap="medium")
    render_engine_card(
        col=up_card_col,
        label=f"Uploaded · {upload_subset}",
        tier=tier,
        true_rul=None,
        cycles=up_cycles,
        ruls=up_ruls,
    )
