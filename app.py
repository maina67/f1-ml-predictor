
# ============================================================================
# F1 RACE PREDICTOR - Complete Web App
# Streamlit app with F1 lights-out intro + Neural Network + GBM ensemble
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os

# ============================================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================================

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# GLOBAL CSS
# ============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;900&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
}

.stApp {
    background: #080810;
    color: #e8e8f0;
}

/* ── Lights-out overlay ── */
#f1-intro {
    position: fixed;
    inset: 0;
    background: #04040a;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 99999;
    gap: 48px;
}

.intro-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2.5rem, 6vw, 5rem);
    font-weight: 900;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #fff;
    opacity: 0;
    animation: fadeSlideUp 0.6s ease 0.2s forwards;
}

.intro-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5em;
    text-transform: uppercase;
    color: #666;
    opacity: 0;
    animation: fadeSlideUp 0.6s ease 0.5s forwards;
    margin-top: -36px;
}

.lights-row {
    display: flex;
    gap: 28px;
    align-items: center;
}

.light-pod {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    background: #111;
    border: 3px solid #222;
    box-shadow: inset 0 0 12px rgba(0,0,0,0.8);
    transition: background 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

.light-pod.red {
    background: #cc0000;
    border-color: #ff2020;
    box-shadow:
        0 0 18px 6px rgba(220,0,0,0.7),
        0 0 40px 12px rgba(180,0,0,0.4),
        inset 0 0 20px rgba(255,80,80,0.4);
}

.light-pod.go {
    background: #003300;
    border-color: #00cc44;
    box-shadow:
        0 0 18px 6px rgba(0,200,60,0.7),
        0 0 40px 12px rgba(0,160,40,0.4),
        inset 0 0 20px rgba(40,255,100,0.3);
    animation: greenPulse 0.4s ease;
}

@keyframes greenPulse {
    0%   { transform: scale(1.15); }
    100% { transform: scale(1); }
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid #1e1e2e;
}

[data-testid="stSidebar"] * {
    color: #c8c8d8 !important;
}

/* ── Main header ── */
.page-header {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 900;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #fff;
    margin: 0;
    line-height: 1;
}

.page-header span {
    color: #e10600;
}

.header-rule {
    height: 3px;
    background: linear-gradient(90deg, #e10600 0%, #ff6b35 40%, transparent 100%);
    border: none;
    margin: 8px 0 28px;
}

/* ── Metric cards ── */
.metric-card {
    background: #0f0f1e;
    border: 1px solid #1e1e34;
    border-radius: 8px;
    padding: 18px 20px;
    transition: border-color 0.2s;
}

.metric-card:hover {
    border-color: #e10600;
}

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}

.metric-value.red { color: #e10600; }
.metric-value.green { color: #00cc66; }
.metric-value.gold { color: #f5a623; }

/* ── Podium ── */
.podium-p1 {
    background: linear-gradient(135deg, #b8860b 0%, #ffd700 100%);
    color: #000;
    padding: 14px 18px;
    border-radius: 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 5px 0;
}

.podium-p2 {
    background: linear-gradient(135deg, #666 0%, #bbb 100%);
    color: #000;
    padding: 12px 18px;
    border-radius: 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 5px 0;
}

.podium-p3 {
    background: linear-gradient(135deg, #7a4312 0%, #cd7f32 100%);
    color: #fff;
    padding: 12px 18px;
    border-radius: 8px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 5px 0;
}

/* ── Racing stripe ── */
.stripe {
    height: 2px;
    background: linear-gradient(90deg, #e10600, #ff6b35 30%, #1e1e2e 70%);
    margin: 20px 0;
    border-radius: 2px;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #0c0c18 !important;
    border-bottom: 1px solid #1e1e2e !important;
    gap: 4px !important;
}

[data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-size: 0.9rem !important;
    color: #666 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 20px !important;
}

[aria-selected="true"] {
    color: #fff !important;
    border-bottom: 2px solid #e10600 !important;
    background: transparent !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #e10600 !important;
    color: #fff !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.65rem 1.5rem !important;
    transition: background 0.2s !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #b80500 !important;
}

/* ── Selectbox / number input ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #0f0f1e !important;
    border-color: #1e1e34 !important;
    color: #e8e8f0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #0f0f1e;
    border-radius: 8px;
    border: 1px solid #1e1e34;
}

/* ── Info / success / warning ── */
[data-testid="stAlert"] {
    background: #0f0f1e !important;
    border-radius: 8px !important;
}

/* ── Section headers ── */
.section-head {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #e10600;
    margin: 20px 0 10px;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin: 2px;
}

.badge-gbm   { background: #1a2a3a; color: #4db8ff; border: 1px solid #4db8ff33; }
.badge-nn    { background: #2a1a2a; color: #cc44ff; border: 1px solid #cc44ff33; }
.badge-ens   { background: #2a2a1a; color: #ffcc44; border: 1px solid #ffcc4433; }
.badge-best  { background: #2a1a1a; color: #ff4444; border: 1px solid #ff444433; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# F1 LIGHTS-OUT INTRO
# ============================================================================

def run_lights_out_intro():
    """
    Renders the F1 starting lights sequence as a Streamlit overlay.
    Shows 5 red lights coming on one by one, then all go off together.
    """
    intro = st.empty()

    def render(states, go=False):
        lights_html = ""
        for s in states:
            cls = "go" if go and s else ("red" if s else "")
            lights_html += f'<div class="light-pod {cls}"></div>'

        intro.markdown(f"""
        <div id="f1-intro">
            <div class="intro-title">F1 <span style="color:#e10600">Race</span> Predictor</div>
            <div class="intro-sub">Powered by Machine Learning</div>
            <div class="lights-row">{lights_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # All lights off
    render([False, False, False, False, False])
    time.sleep(0.8)

    # Lights come on one by one
    for i in range(1, 6):
        render([j < i for j in range(5)])
        time.sleep(0.75)

    # Hold all red
    time.sleep(1.4)

    # LIGHTS OUT — all go green briefly
    render([True, True, True, True, True], go=True)
    time.sleep(0.7)

    # Clear intro
    intro.empty()


# Only run intro once per session
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False

if not st.session_state.intro_done:
    run_lights_out_intro()
    st.session_state.intro_done = True
    st.rerun()


# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

@st.cache_resource(show_spinner="Loading models...")
def load_all():
    data_dir = "data"

    # Model info
    with open(f"{data_dir}/model_info.json") as f:
        info = json.load(f)

    feature_columns = info["feature_columns"]

    # GBM model (always available)
    gbm = joblib.load(f"{data_dir}/f1_best_model.pkl")

    # Neural network (optional — may not exist yet)
    nn = None
    scaler = None
    nn_path = f"{data_dir}/f1_neural_network.keras"
    scaler_path = f"{data_dir}/f1_scaler.pkl"

    if os.path.exists(nn_path) and os.path.exists(scaler_path):
        try:
            import tensorflow as tf
            nn = tf.keras.models.load_model(nn_path)
            scaler = joblib.load(scaler_path)
        except Exception:
            nn = None
            scaler = None

    # Historical data
    df = pd.read_csv(f"{data_dir}/f1_data_with_all_features.csv")
    df = df.sort_values(["Year", "Round"]).reset_index(drop=True)

    return gbm, nn, scaler, info, feature_columns, df


try:
    gbm_model, nn_model, scaler, model_info, FEATURE_COLS, df_hist = load_all()
    nn_available = nn_model is not None
except Exception as e:
    st.error(f"❌ Could not load data/models: {e}")
    st.info("Make sure your `data/` folder contains: `f1_best_model.pkl`, `model_info.json`, `f1_data_with_all_features.csv`")
    st.stop()

LATEST_YEAR  = int(df_hist["Year"].max())
LATEST_ROUND = int(df_hist[df_hist["Year"] == LATEST_YEAR]["Round"].max())


# ============================================================================
# PREDICTION HELPERS
# ============================================================================

def get_driver_stats(driver_code):
    rows = df_hist[df_hist["Driver"] == driver_code].tail(5)
    if rows.empty:
        return None
    latest = rows.iloc[-1]
    return {
        "Driver_encoded":             latest["Driver_encoded"],
        "Team":                       latest["Team"],
        "Team_encoded":               latest["Team_encoded"],
        "Driver_Avg_Position_Last5":  rows["FinishPosition"].mean(),
        "Driver_Avg_Position_Last3":  rows["FinishPosition"].tail(3).mean(),
        "Driver_Finish_Rate":         rows["Finished"].mean(),
        "Driver_Race_Count":          int(latest["Driver_Race_Count"]) + 1,
        "Team_Championship_Points":   latest["Team_Championship_Points"],
    }


def build_feature_row(driver_code, quali_pos, grid_pos, circuit_name):
    stats = get_driver_stats(driver_code)
    if stats is None:
        return None

    # Circuit encoding
    circ_rows = df_hist[df_hist["RaceName"] == circuit_name]
    circuit_encoded = int(circ_rows["Circuit_encoded"].iloc[0]) if not circ_rows.empty else 0

    # Team averages
    team = stats["Team"]
    team_rows = df_hist[(df_hist["Year"] == LATEST_YEAR) & (df_hist["Team"] == team)]
    team_avg = team_rows["FinishPosition"].mean() if not team_rows.empty else 10.5
    team_pts = team_rows["Team_Championship_Points"].max() if not team_rows.empty else 0

    # Circuit-specific
    d_circ = df_hist[(df_hist["Driver"] == driver_code) & (df_hist["RaceName"] == circuit_name)]
    t_circ = df_hist[(df_hist["Team"] == team) & (df_hist["RaceName"] == circuit_name)]
    d_circ_avg = d_circ["FinishPosition"].mean() if not d_circ.empty else 10.5
    t_circ_avg = t_circ["FinishPosition"].mean() if not t_circ.empty else 10.5

    # Teammate comparison
    tm_quali = df_hist[(df_hist["Team"] == team) & (df_hist["Year"] == LATEST_YEAR)]["QualiPosition"].mean()

    return {
        "GridPosition":               grid_pos,
        "QualiPosition":              quali_pos,
        "Driver_encoded":             stats["Driver_encoded"],
        "Team_encoded":               stats["Team_encoded"],
        "Circuit_encoded":            circuit_encoded,
        "Driver_Avg_Position_Last5":  stats["Driver_Avg_Position_Last5"],
        "Driver_Avg_Position_Last3":  stats["Driver_Avg_Position_Last3"],
        "Team_Avg_Position_Last5":    team_avg,
        "Driver_Finish_Rate":         stats["Driver_Finish_Rate"],
        "Driver_Circuit_Avg":         d_circ_avg,
        "Team_Circuit_Avg":           t_circ_avg,
        "Quali_Grid_Diff":            quali_pos - grid_pos,
        "Quali_vs_Teammate":          quali_pos - tm_quali,
        "Race_Number_In_Season":      LATEST_ROUND + 1,
        "Driver_Race_Count":          stats["Driver_Race_Count"],
        "Team_Championship_Points":   team_pts,
    }


def predict_race(quali_results, circuit_name, mode="ensemble"):
    """
    mode: 'gbm' | 'nn' | 'ensemble'
    quali_results: list of (driver_code, quali_pos, grid_pos)
    """
    rows, meta = [], []

    for driver_code, quali_pos, grid_pos in quali_results:
        feat = build_feature_row(driver_code, quali_pos, grid_pos, circuit_name)
        if feat is None:
            continue
        rows.append(feat)
        team_info = get_driver_stats(driver_code)
        meta.append({
            "Driver": driver_code,
            "Team":   team_info["Team"] if team_info else "Unknown",
            "Grid":   grid_pos,
            "Quali":  quali_pos,
        })

    if not rows:
        return pd.DataFrame()

    X_df     = pd.DataFrame(rows)[FEATURE_COLS]
    X_scaled = scaler.transform(X_df.values) if scaler is not None else None

    # GBM predictions
    gbm_preds = gbm_model.predict(X_df)

    # NN predictions
    if nn_available and X_scaled is not None:
        nn_preds = nn_model.predict(X_scaled, verbose=0).flatten()
    else:
        nn_preds = gbm_preds.copy()

    # Choose output
    if mode == "gbm" or not nn_available:
        final_preds = gbm_preds
    elif mode == "nn":
        final_preds = nn_preds
    else:  # ensemble
        final_preds = 0.5 * gbm_preds + 0.5 * nn_preds

    results = []
    for i, m in enumerate(meta):
        results.append({
            **m,
            "GBM_Pred":   round(float(gbm_preds[i]), 1),
            "NN_Pred":    round(float(nn_preds[i]),  1),
            "Predicted":  round(float(final_preds[i]), 1),
        })

    df_out = pd.DataFrame(results).sort_values("Predicted").reset_index(drop=True)
    df_out["Rank"] = range(1, len(df_out) + 1)
    return df_out


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.6rem;
                font-weight:900; letter-spacing:0.15em; text-transform:uppercase;
                color:#fff; margin-bottom:4px;">
        🏎️ F1 PREDICTOR
    </div>
    <div style="font-size:0.7rem; letter-spacing:0.2em; color:#555;
                text-transform:uppercase; margin-bottom:16px;">
        ML Race Analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

    # Model status
    st.markdown('<div class="section-head">Models Active</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="model-badge badge-gbm">✓ Gradient Boosting</span>', unsafe_allow_html=True)
    if nn_available:
        st.markdown(f'<span class="model-badge badge-nn">✓ Neural Network</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="model-badge badge-ens">✓ Ensemble</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="model-badge" style="background:#1a1a1a;color:#444;border:1px solid #333;">⏳ Neural Network (pending)</span>', unsafe_allow_html=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

    # Performance
    st.markdown('<div class="section-head">Performance</div>', unsafe_allow_html=True)

    gbm_mae = model_info.get("gbm_model", {}).get("test_mae", model_info.get("test_mae", "—"))
    nn_mae  = model_info.get("nn_model",  {}).get("test_mae", "—")
    ens_mae = model_info.get("ensemble",  {}).get("test_mae", "—")

    for label, val, cls in [
        ("GBM MAE",      gbm_mae, "red"),
        ("NN MAE",       nn_mae,  "green" if isinstance(nn_mae, float) and isinstance(gbm_mae, float) and nn_mae < gbm_mae else "red"),
        ("Ensemble MAE", ens_mae, "gold"),
    ]:
        val_str = f"{val:.3f} pos" if isinstance(val, float) else str(val)
        st.markdown(f"""
        <div class="metric-card" style="margin:6px 0; padding:12px 16px;">
            <div class="metric-label">{label}</div>
            <div class="metric-value {cls}">{val_str}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Data Coverage</div>
        <div class="metric-value" style="font-size:1.2rem; color:#aaa;">
            2021 – {LATEST_YEAR}
        </div>
        <div style="font-size:0.75rem; color:#444; margin-top:4px;">
            Latest: Round {LATEST_ROUND}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.markdown("""
<div class="page-header">🏁 F1 <span>Race</span> Predictor</div>
<hr class="header-rule">
""", unsafe_allow_html=True)

tab_predict, tab_analysis, tab_about = st.tabs([
    "🔮  PREDICT RACE",
    "📊  MODEL ANALYSIS",
    "ℹ️  ABOUT",
])


# ============================================================================
# TAB 1 — PREDICT RACE
# ============================================================================

with tab_predict:
    col_setup, col_results = st.columns([1, 1], gap="large")

    # ── Left: Setup ──────────────────────────────────────────────────────────
    with col_setup:
        st.markdown('<div class="section-head">Race Setup</div>', unsafe_allow_html=True)

        # Circuit
        circuits = sorted(df_hist["RaceName"].unique())
        circuit = st.selectbox("🏁 Circuit", circuits)

        # Prediction mode
        mode_options = ["Ensemble (GBM + NN)", "Gradient Boosting only", "Neural Network only"]
        if not nn_available:
            mode_options = ["Gradient Boosting only"]
            st.info("💡 Train the neural network in Colab to unlock Ensemble and NN modes.")
        mode_label = st.selectbox("🤖 Prediction Mode", mode_options)
        mode = "ensemble" if "Ensemble" in mode_label else ("nn" if "Neural" in mode_label else "gbm")

        st.markdown('<div class="section-head" style="margin-top:20px;">Qualifying Order</div>', unsafe_allow_html=True)
        st.caption("Set each driver's qualifying position and any grid penalties.")

        # Driver list from latest season
        active_drivers = sorted(df_hist[df_hist["Year"] == LATEST_YEAR]["Driver"].unique())

        quali_data = []
        for i in range(20):
            c1, c2, c3 = st.columns([3, 1, 1])
            default_idx = min(i, len(active_drivers) - 1)
            with c1:
                drv = st.selectbox(
                    f"P{i+1:02d}",
                    active_drivers,
                    index=default_idx,
                    key=f"drv_{i}",
                    label_visibility="collapsed",
                )
            with c2:
                st.markdown(
                    f"<div style='padding:8px 0; color:#aaa; font-size:0.9rem;'>Q: P{i+1}</div>",
                    unsafe_allow_html=True
                )
            with c3:
                grid = st.number_input(
                    "Grid",
                    min_value=1, max_value=20,
                    value=i + 1,
                    key=f"grid_{i}",
                    label_visibility="collapsed",
                )
            quali_data.append((drv, i + 1, grid))

        predict_btn = st.button("🚦 PREDICT RACE RESULTS")

    # ── Right: Results ───────────────────────────────────────────────────────
    with col_results:
        st.markdown('<div class="section-head">Predicted Results</div>', unsafe_allow_html=True)

        if predict_btn or "last_results" in st.session_state:
            if predict_btn:
                with st.spinner("Running prediction models..."):
                    results = predict_race(quali_data, circuit, mode)
                    st.session_state["last_results"] = results
                    st.session_state["last_circuit"] = circuit
                    st.session_state["last_mode"]    = mode_label
            else:
                results = st.session_state["last_results"]

            if not results.empty:
                st.success(f"**{st.session_state['last_circuit']}** · {st.session_state['last_mode']}")

                # Podium
                st.markdown('<div class="section-head">Podium</div>', unsafe_allow_html=True)
                podium_classes = ["podium-p1", "podium-p2", "podium-p3"]
                medals = ["🥇", "🥈", "🥉"]
                for j in range(min(3, len(results))):
                    row = results.iloc[j]
                    st.markdown(
                        f'<div class="{podium_classes[j]}">'
                        f'{medals[j]} {row["Driver"]} &nbsp;·&nbsp; '
                        f'<span style="opacity:0.7;font-weight:400">{row["Team"]}</span>'
                        f'&nbsp; — &nbsp;Grid P{row["Grid"]} → Pred P{row["Predicted"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                # Full table
                st.markdown('<div class="section-head" style="margin-top:20px;">Full Classification</div>', unsafe_allow_html=True)
                display_cols = ["Rank", "Driver", "Team", "Grid", "Predicted"]
                if nn_available:
                    display_cols += ["GBM_Pred", "NN_Pred"]
                st.dataframe(
                    results[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=results["Driver"],
                    y=results["Predicted"],
                    marker=dict(
                        color=results["Predicted"],
                        colorscale=[[0, "#e10600"], [0.5, "#ff6b35"], [1, "#2a2a4a"]],
                        showscale=False,
                    ),
                    text=results["Predicted"].apply(lambda x: f"P{x:.0f}"),
                    textposition="outside",
                ))
                fig.update_layout(
                    paper_bgcolor="#080810",
                    plot_bgcolor="#0c0c18",
                    font=dict(color="#e8e8f0", family="Barlow Condensed"),
                    title=dict(text="Predicted Finishing Positions", font=dict(size=16)),
                    xaxis=dict(gridcolor="#1e1e2e", title=""),
                    yaxis=dict(gridcolor="#1e1e2e", title="Position", autorange="reversed"),
                    height=380,
                    margin=dict(t=50, b=20, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv = results.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name=f"f1_prediction_{circuit.replace(' ', '_')}.csv",
                    mime="text/csv",
                )
        else:
            st.markdown("""
            <div style="color:#444; text-align:center; padding:60px 20px;
                        border:1px dashed #1e1e2e; border-radius:8px; margin-top:40px;">
                <div style="font-size:3rem;">🏁</div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-size:1.2rem;
                            letter-spacing:0.1em; text-transform:uppercase; margin-top:10px;">
                    Set up qualifying results<br>and hit Predict
                </div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# TAB 2 — MODEL ANALYSIS
# ============================================================================

with tab_analysis:
    st.markdown('<div class="section-head">Model Performance Comparison</div>', unsafe_allow_html=True)

    # Performance cards
    c1, c2, c3 = st.columns(3)
    perf_data = [
        ("Gradient Boosting", gbm_mae, "red",   "badge-gbm"),
        ("Neural Network",    nn_mae,  "green",  "badge-nn"),
        ("Ensemble",          ens_mae, "gold",   "badge-ens"),
    ]
    for col, (name, mae, cls, badge) in zip([c1, c2, c3], perf_data):
        val_str = f"{mae:.3f}" if isinstance(mae, float) else "Pending"
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <span class="model-badge {badge}">{name}</span>
                <div class="metric-label" style="margin-top:10px;">Test MAE</div>
                <div class="metric-value {cls}">{val_str}</div>
                <div style="font-size:0.75rem; color:#555; margin-top:4px;">positions avg error</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

    # Feature importance
    col_feat, col_data = st.columns([1, 1], gap="large")

    with col_feat:
        st.markdown('<div class="section-head">Feature Importance (GBM)</div>', unsafe_allow_html=True)

        if hasattr(gbm_model, "feature_importances_"):
            imp_df = pd.DataFrame({
                "Feature":    FEATURE_COLS,
                "Importance": gbm_model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(12)

            fig_imp = go.Figure(go.Bar(
                x=imp_df["Importance"],
                y=imp_df["Feature"],
                orientation="h",
                marker=dict(
                    color=imp_df["Importance"],
                    colorscale=[[0, "#1e1e2e"], [1, "#e10600"]],
                    showscale=False,
                ),
                text=imp_df["Importance"].apply(lambda x: f"{x:.1%}"),
                textposition="outside",
            ))
            fig_imp.update_layout(
                paper_bgcolor="#080810",
                plot_bgcolor="#0c0c18",
                font=dict(color="#e8e8f0", family="Barlow Condensed"),
                xaxis=dict(gridcolor="#1e1e2e", showticklabels=False),
                yaxis=dict(gridcolor="#1e1e2e"),
                height=380,
                margin=dict(t=10, b=10, l=10, r=60),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_data:
        st.markdown('<div class="section-head">Training Data by Year</div>', unsafe_allow_html=True)

        year_summary = df_hist.groupby("Year").agg(
            Races=("Round", "nunique"),
            Records=("Driver", "count"),
        ).reset_index()

        fig_yr = go.Figure(go.Bar(
            x=year_summary["Year"].astype(str),
            y=year_summary["Races"],
            marker=dict(
                color=year_summary["Races"],
                colorscale=[[0, "#1e1e2e"], [1, "#e10600"]],
                showscale=False,
            ),
            text=year_summary["Races"],
            textposition="outside",
        ))
        fig_yr.update_layout(
            paper_bgcolor="#080810",
            plot_bgcolor="#0c0c18",
            font=dict(color="#e8e8f0", family="Barlow Condensed"),
            xaxis=dict(gridcolor="#1e1e2e"),
            yaxis=dict(gridcolor="#1e1e2e", title="Races"),
            height=200,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_yr, use_container_width=True)

        # Summary table
        st.dataframe(year_summary, use_container_width=True, hide_index=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

    # What the MAE means in context
    st.markdown('<div class="section-head">What Does MAE Mean in F1?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">MAE = 3.0 means...</div>
            <div style="margin-top:8px; color:#aaa; line-height:1.7;">
                Predict <strong style="color:#fff">P5</strong> → actual result is usually <strong style="color:#e10600">P2–P8</strong><br>
                Predict <strong style="color:#fff">P1</strong> → actual result is usually <strong style="color:#e10600">P1–P4</strong><br>
                Predict <strong style="color:#fff">P15</strong> → actual result is usually <strong style="color:#e10600">P12–P18</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Why not 100% accurate?</div>
            <div style="margin-top:8px; color:#aaa; line-height:1.7;">
                🌧️ <strong style="color:#fff">Weather</strong> changes race outcomes<br>
                💥 <strong style="color:#fff">Crashes & Safety Cars</strong> are random<br>
                🔧 <strong style="color:#fff">Pit strategy</strong> calls are unpredictable<br>
                🏎️ <strong style="color:#fff">Mechanical failures</strong> can't be forecast
            </div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# TAB 3 — ABOUT
# ============================================================================

with tab_about:
    st.markdown('<div class="section-head">How It Works</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Data Collection",   "FastF1 API pulls official F1 telemetry, qualifying & race results for 2021–2026"),
        ("02", "Feature Engineering","16 smart features: rolling form, circuit-specific stats, qualifying advantage, team strength"),
        ("03", "Model Training",     "Gradient Boosting + Neural Network trained on 2,000+ race records"),
        ("04", "Ensemble Prediction","Both models combined for the most accurate prediction"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="metric-card" style="display:flex; gap:20px; align-items:flex-start; margin:8px 0;">
            <div style="font-family:'Barlow Condensed',sans-serif; font-size:2rem;
                        font-weight:900; color:#e10600; min-width:40px;">{num}</div>
            <div>
                <div style="font-family:'Barlow Condensed',sans-serif; font-weight:700;
                            font-size:1.1rem; letter-spacing:0.05em;">{title}</div>
                <div style="color:#666; font-size:0.9rem; margin-top:4px;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="stripe"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-head">Tech Stack</div>', unsafe_allow_html=True)
        for tech, detail in [
            ("FastF1",      "Official F1 telemetry API"),
            ("scikit-learn","Gradient Boosting model"),
            ("TensorFlow",  "Neural Network model"),
            ("Streamlit",   "Web interface"),
            ("Plotly",      "Interactive charts"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:8px 0;
                        border-bottom:1px solid #1e1e2e; font-size:0.9rem;">
                <span style="color:#e10600; font-weight:600;">{tech}</span>
                <span style="color:#666;">{detail}</span>
            </div>
            """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-head">Limitations</div>', unsafe_allow_html=True)
        for lim in [
            "Cannot predict crashes or safety cars",
            "Weather impact is not modelled",
            "New regulations can change car hierarchy",
            "New drivers have limited historical data",
            "Pit stop strategies are not considered",
        ]:
            st.markdown(f"""
            <div style="padding:8px 0; border-bottom:1px solid #1e1e2e;
                        font-size:0.9rem; color:#666;">
                ⚠️ &nbsp; {lim}
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; color:#333; font-size:0.75rem;
            letter-spacing:0.1em; text-transform:uppercase; margin-top:40px;
            padding-top:20px; border-top:1px solid #1e1e2e;">
    F1 Race Predictor &nbsp;·&nbsp; Data: 2021–{LATEST_YEAR} &nbsp;·&nbsp; {datetime.now().strftime("%Y")}
</div>
""", unsafe_allow_html=True)
