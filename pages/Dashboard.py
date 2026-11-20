"""
Page 1 — Dashboard
Ringkasan KPI, perbandingan model, dan tren PDRB per provinsi.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import (
    load_dataset, load_projections,
    PROVINSI_LIST, PROV_COLORS, TARGET,
)

st.set_page_config(page_title="Dashboard · PDRB Jawa", page_icon="🏠", layout="wide")

# ── Global CSS (sama dengan app.py) ───────────────────────────────────────
st.markdown("""
<style>
.accent-bar { height:4px; background:linear-gradient(90deg,#0A6E8A,#12B4D2);
              border-radius:2px; margin-bottom:1.2rem; }
.kpi-card { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;
            padding:1rem 1.2rem; }
.kpi-label { font-size:.72rem; font-weight:600; letter-spacing:.06em;
             color:#6B7280; text-transform:uppercase; margin-bottom:.2rem; }
.kpi-value { font-size:1.9rem; font-weight:700; color:#0F172A; line-height:1.1; }
.kpi-delta { font-size:.8rem; margin-top:.3rem; }
.delta-pos { color:#059669; } .delta-neg { color:#DC2626; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.title("🏠 Dashboard")
st.caption("Ringkasan performa model dan kondisi pertumbuhan ekonomi Pulau Jawa")

# ── Load data ──────────────────────────────────────────────────────────────
df      = load_dataset()
df_proj = load_projections()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI Cards
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Performa Model Terbaik · Naive+XGB_strict (a=1.0)")

metrics = {
    "Theil's U Statistic":  ("0.880", "< 1.0 → lebih baik dari Naive", True),
    "MAPE (Test Set)":      ("6.11%",  "threshold KPI: < 6%",           None),
    "RMSE (Test Set)":      ("0.3797", "satuan: % YoY",                 True),
    "Hit Rate (Arah)":      ("61.5%",  "threshold: > 55%",              True),
    "DM-Test p-value":      ("0.031",  "signifikan vs ARIMA baseline",  True),
    "Backtesting MAPE":     ("2.81%",  "realisasi 2025–2026",           True),
}

cols = st.columns(len(metrics))
for col, (label, (val, note, good)) in zip(cols, metrics.items()):
    delta_cls = "delta-pos" if good is True else ("delta-neg" if good is False else "")
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-delta <span class='{delta_cls}'>{note}</span>"></div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Perbandingan Model (leaderboard)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Leaderboard Perbandingan Model")

model_data = pd.DataFrame([
    {"Model": "Naive+XGB_strict (SW+Reg, a=1.0)", "RMSE": 0.3797, "MAPE (%)": 6.11,  "Theil's U": 0.880, "Hit Rate (%)": 61.5, "DM p-val": 0.031},
    {"Model": "Ensemble A (RF+XGB, 50/50)",         "RMSE": 0.3812, "MAPE (%)": 6.22,  "Theil's U": 0.885, "Hit Rate (%)": 59.6, "DM p-val": 0.038},
    {"Model": "Ensemble B",                          "RMSE": 0.3850, "MAPE (%)": 6.44,  "Theil's U": 0.885, "Hit Rate (%)": 59.6, "DM p-val": 0.040},
    {"Model": "Naive+RF_strict",                     "RMSE": 0.3904, "MAPE (%)": 6.78,  "Theil's U": 0.892, "Hit Rate (%)": 57.7, "DM p-val": 0.044},
    {"Model": "Ensemble C (Stacked Ridge)",          "RMSE": 0.3990, "MAPE (%)": 7.01,  "Theil's U": 0.911, "Hit Rate (%)": 55.8, "DM p-val": 0.051},
    {"Model": "XGBoost Standalone",                  "RMSE": 0.4150, "MAPE (%)": 7.80,  "Theil's U": 1.012, "Hit Rate (%)": 51.9, "DM p-val": 0.312},
    {"Model": "Random Forest Standalone",            "RMSE": 0.4280, "MAPE (%)": 8.20,  "Theil's U": 1.028, "Hit Rate (%)": 50.0, "DM p-val": 0.410},
    {"Model": "ARIMA",                               "RMSE": 0.4316, "MAPE (%)": 8.55,  "Theil's U": 1.048, "Hit Rate (%)": 48.1, "DM p-val": None},
    {"Model": "VAR",                                 "RMSE": 0.4490, "MAPE (%)": 9.10,  "Theil's U": 1.090, "Hit Rate (%)": 46.2, "DM p-val": None},
    {"Model": "ANN (MLP)",                           "RMSE": 0.4620, "MAPE (%)": 9.80,  "Theil's U": 1.121, "Hit Rate (%)": 44.2, "DM p-val": None},
    {"Model": "Naive Random Walk (Baseline)",        "RMSE": 0.4895, "MAPE (%)": 10.50, "Theil's U": 1.000, "Hit Rate (%)": 50.0, "DM p-val": None},
])

# Highlight baris terbaik
def highlight_best(row):
    if "Naive+XGB_strict" in row["Model"]:
        return ["background-color:#D1FAE5; font-weight:600"] * len(row)
    elif row["Theil's U"] < 1.0:
        return ["background-color:#EFF6FF"] * len(row)
    elif row["Theil's U"] > 1.0:
        return ["background-color:#FEF2F2; color:#9CA3AF"] * len(row)
    return [""] * len(row)

st.dataframe(
    model_data.style.apply(highlight_best, axis=1).format({
        "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "Theil's U": "{:.3f}",
        "Hit Rate (%)": "{:.1f}", "DM p-val": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption("🟢 Hijau = model terbaik · 🔵 Biru = Theil's U < 1.0 (lebih baik dari naive) · 🔴 Merah = di bawah baseline")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Bar chart Theil's U
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Theil's U Statistic per Model")

fig_u = go.Figure()
colors_bar = [
    "#0A6E8A" if "Naive+XGB_strict" in m else
    ("#64B5F6" if u < 1.0 else "#EF9A9A")
    for m, u in zip(model_data["Model"], model_data["Theil's U"])
]

fig_u.add_trace(go.Bar(
    x=model_data["Theil's U"],
    y=model_data["Model"],
    orientation="h",
    marker_color=colors_bar,
    text=[f"{u:.3f}" for u in model_data["Theil's U"]],
    textposition="outside",
))
fig_u.add_vline(x=1.0, line_dash="dash", line_color="#DC2626",
                annotation_text="Baseline Naive (U=1.0)",
                annotation_position="top right")
fig_u.update_layout(
    height=420, margin=dict(l=10, r=60, t=20, b=20),
    xaxis_title="Theil's U (lebih kecil = lebih baik)",
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(size=11),
)
fig_u.update_xaxes(range=[0, 1.3], gridcolor="#F1F5F9")
st.plotly_chart(fig_u, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Tren PDRB Historis (semua provinsi)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Tren PDRB Growth Historis per Provinsi (2011–2024)")

df_hist = df[df["Tahun"] <= 2024].copy()
df_hist["Period_dt"] = pd.to_datetime(
    df_hist["Tahun"].astype(str) + df_hist["Kuartal"].map({"Q1":"01","Q2":"04","Q3":"07","Q4":"10"}),
    format="%Y%m"
)

col_left, col_right = st.columns([3, 1])
with col_right:
    prov_sel = st.multiselect(
        "Tampilkan provinsi:",
        PROVINSI_LIST,
        default=PROVINSI_LIST,
        key="dash_prov",
    )
    show_covid = st.checkbox("Tandai COVID Q2 2020", value=True)

with col_left:
    fig_hist = go.Figure()
    for prov in prov_sel:
        sub = df_hist[df_hist["Provinsi"] == prov].sort_values("Period_dt")
        fig_hist.add_trace(go.Scatter(
            x=sub["Period_dt"], y=sub[TARGET],
            mode="lines+markers",
            name=prov,
            line=dict(color=PROV_COLORS[prov], width=1.8),
            marker=dict(size=3),
        ))

    if show_covid:
        fig_hist.add_vline(
            x="2020-04-01", line_dash="dot", line_color="#DC2626",
            annotation_text="COVID Q2-2020", annotation_position="top right",
        )
    fig_hist.add_hline(y=0, line_color="#6B7280", line_width=0.8)
    fig_hist.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=20),
        yaxis_title="PDRB Growth YoY (%)",
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11),
    )
    fig_hist.update_xaxes(gridcolor="#F1F5F9")
    fig_hist.update_yaxes(gridcolor="#F1F5F9")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Proyeksi ringkasan (jika ada pkl)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### Proyeksi PDRB Growth 2025–2027 · Naive+XGB_strict")

if df_proj is not None:
    piv = df_proj.pivot_table(
        index=["Tahun", "Kuartal"], columns="Provinsi", values="XGB_strict"
    ).round(2)
    st.dataframe(piv, use_container_width=True)
else:
    st.info(
        "File `assets/df_proj_out.pkl` belum tersedia.  \n"
        "Jalankan notebook hingga Bab 8, lalu:  \n"
        "```python\nimport joblib\njoblib.dump(df_proj_out, 'assets/df_proj_out.pkl')\n```"
    )
