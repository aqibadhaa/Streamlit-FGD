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

st.set_page_config(page_title="Dashboard · PDRB Jawa", page_icon="Dashboard", layout="wide")

# -- Global CSS ---------------------------------------------------------------
st.markdown("""
<style>
.accent-bar { height:3px; background:#B8CCE0;
              border-radius:2px; margin-bottom:1.2rem; }
.kpi-card { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px;
            padding:1rem 1.2rem; }
.kpi-label { font-size:.72rem; font-weight:600; letter-spacing:.06em;
             color:#718096; text-transform:uppercase; margin-bottom:.2rem; }
.kpi-value { font-size:1.9rem; font-weight:700; color:#2D3748; line-height:1.1; }
.kpi-delta { font-size:.8rem; margin-top:.3rem; }
.delta-pos { color:#2F855A; } .delta-neg { color:#C53030; }

/* Leaderboard section headers */
.lb-header-blue {
    background:#EBF4FA; border-left:4px solid #5B9EC9;
    border-radius:0 6px 6px 0; padding:.5rem .9rem;
    margin-bottom:.5rem;
}
.lb-header-red {
    background:#FDF0F0; border-left:4px solid #C47E7E;
    border-radius:0 6px 6px 0; padding:.5rem .9rem;
    margin-bottom:.5rem;
}
.lb-title { font-size:.95rem; font-weight:700; color:#2D3748; }
.lb-sub   { font-size:.75rem; color:#718096; margin-top:.1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.title("Dashboard")
st.caption("Ringkasan performa model dan kondisi pertumbuhan ekonomi Pulau Jawa")

# -- Load data ----------------------------------------------------------------
df      = load_dataset()
df_proj = load_projections()

# ===========================================================================
# SECTION 1 -- KPI Cards
# ===========================================================================
st.markdown("#### Performa Model Terbaik · Naive+XGB_strict (a=1.0)")

metrics = {
    "Theil's U Statistic":  ("0.880", "< 1.0 → lebih baik dari Naive", True),
    "MAPE (Test Set)":      ("6.11%",  "threshold KPI: < 6%",           None),
    "RMSE (Test Set)":      ("0.3797", "satuan: % YoY",                 True),
    "Hit Rate (Arah)":      ("61.5%",  "threshold: > 55%",              True),
    "DM-Test p-value":      ("0.031",  "signifikan vs ARIMA baseline",  True),
    "Backtesting MAPE":     ("2.81%",  "realisasi 2025-2026",           True),
}

cols = st.columns(len(metrics))
for col, (label, (val, note, good)) in zip(cols, metrics.items()):
    delta_cls = "delta-pos" if good is True else ("delta-neg" if good is False else "")
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-delta"><span class="{delta_cls}">{note}</span></div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ===========================================================================
# SECTION 2 -- Leaderboard Perbandingan Model (3 bagian)
# ===========================================================================
st.markdown("#### Leaderboard Perbandingan Model")

# ---------------------------------------------------------------------------
# Data model: Siklus 1 (Standalone) dan Siklus 2 (Hibrida/Kombinasi)
# ---------------------------------------------------------------------------

SIKLUS_1 = pd.DataFrame([
    {"Model": "XGBoost Standalone",        "RMSE": 0.4150, "MAPE (%)": 7.80, "Theil's U": 1.012, "Hit Rate (%)": 51.9, "DM p-val": 0.312},
    {"Model": "Random Forest Standalone",  "RMSE": 0.4280, "MAPE (%)": 8.20, "Theil's U": 1.028, "Hit Rate (%)": 50.0, "DM p-val": 0.410},
    {"Model": "ARIMA",                     "RMSE": 0.4316, "MAPE (%)": 8.55, "Theil's U": 1.048, "Hit Rate (%)": 48.1, "DM p-val": None},
    {"Model": "VAR",                       "RMSE": 0.4490, "MAPE (%)": 9.10, "Theil's U": 1.090, "Hit Rate (%)": 46.2, "DM p-val": None},
    {"Model": "ANN (MLP)",                 "RMSE": 0.4620, "MAPE (%)": 9.80, "Theil's U": 1.121, "Hit Rate (%)": 44.2, "DM p-val": None},
])

SIKLUS_2 = pd.DataFrame([
    {"Model": "Naive+XGB_strict (SW+Reg, a=1.0)", "RMSE": 0.3797, "MAPE (%)": 6.11, "Theil's U": 0.880, "Hit Rate (%)": 61.5, "DM p-val": 0.031},
    {"Model": "Ensemble A (RF+XGB, 50/50)",        "RMSE": 0.3812, "MAPE (%)": 6.22, "Theil's U": 0.885, "Hit Rate (%)": 59.6, "DM p-val": 0.038},
    {"Model": "Ensemble B",                         "RMSE": 0.3850, "MAPE (%)": 6.44, "Theil's U": 0.885, "Hit Rate (%)": 59.6, "DM p-val": 0.040},
    {"Model": "Naive+RF_strict",                    "RMSE": 0.3904, "MAPE (%)": 6.78, "Theil's U": 0.892, "Hit Rate (%)": 57.7, "DM p-val": 0.044},
    {"Model": "Ensemble C (Stacked Ridge)",         "RMSE": 0.3990, "MAPE (%)": 7.01, "Theil's U": 0.911, "Hit Rate (%)": 55.8, "DM p-val": 0.051},
])

fmt_dict = {
    "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "Theil's U": "{:.3f}",
    "Hit Rate (%)": "{:.1f}",
    "DM p-val": lambda x: f"{x:.3f}" if pd.notna(x) else "-",
}

# -- Tabel 1: Siklus 1 (Standalone) — tema biru soft pastel ------------------
def highlight_siklus1(row):
    # Model terbaik di Siklus 1 = Theil's U terkecil = XGBoost Standalone
    if "XGBoost Standalone" in row["Model"]:
        # Aksen biru lebih muda (highlight terbaik dalam siklus ini)
        return ["background-color:#D6EAF8; font-weight:700; color:#1A5276"] * len(row)
    return ["background-color:#EBF5FB; color:#2C3E50"] * len(row)

st.markdown("""
<div class="lb-header-blue">
  <div class="lb-title">Siklus 1 — Model Standalone</div>
  <div class="lb-sub">Model individual tanpa kombinasi · XGBoost, RF, ARIMA, VAR, ANN</div>
</div>
""", unsafe_allow_html=True)

st.dataframe(
    SIKLUS_1.style.apply(highlight_siklus1, axis=1).format(fmt_dict),
    use_container_width=True,
    hide_index=True,
)
st.caption("Biru muda = semua model Siklus 1 · Biru lebih cerah = terbaik dalam siklus ini (Theil's U terkecil)")

# -- Tabel 2: Siklus 2 (Hibrida/Kombinasi) — tema merah burgundy muda --------
def highlight_siklus2(row):
    # Model terbaik di Siklus 2 = Naive+XGB_strict
    if "Naive+XGB_strict" in row["Model"]:
        # Aksen merah lebih muda (highlight terbaik dalam siklus ini)
        return ["background-color:#FADBD8; font-weight:700; color:#922B21"] * len(row)
    return ["background-color:#FDEDEC; color:#2C3E50"] * len(row)

st.markdown("""
<div class="lb-header-red">
  <div class="lb-title">Siklus 2 — Model Kombinasi Hibrida</div>
  <div class="lb-sub">Naive+ dan Ensemble · Naive+XGB, Naive+RF, Ensemble A/B/C</div>
</div>
""", unsafe_allow_html=True)

st.dataframe(
    SIKLUS_2.style.apply(highlight_siklus2, axis=1).format(fmt_dict),
    use_container_width=True,
    hide_index=True,
)
st.caption("Merah muda = semua model Siklus 2 · Merah lebih cerah = terbaik dalam siklus ini (Naive+XGB_strict)")

st.divider()

# ===========================================================================
# SECTION 3 -- Bar chart Theil's U (semua model, sorted asc, warna per siklus)
# ===========================================================================
st.markdown("#### Theil's U Statistic per Model")
st.caption("Semua model diurutkan dari Theil's U terkecil. Biru = Standalone (Siklus 1) · Merah muda = Hibrida (Siklus 2)")

# Gabung semua model dengan label siklus, sort by Theil's U ascending
all_models = pd.concat([
    SIKLUS_2.assign(Siklus="Hibrida"),
    SIKLUS_1.assign(Siklus="Standalone"),
], ignore_index=True).sort_values("Theil's U", ascending=True)

# Warna: hibrida = merah muda muted, standalone = biru muted
# Model terbaik overall (Naive+XGB_strict) dapat aksen sedikit lebih gelap
def bar_color(row):
    if "Naive+XGB_strict" in row["Model"]:
        return "#B07A8A"   # burgundy muted — terbaik overall
    elif row["Siklus"] == "Hibrida":
        return "#D4A0A8"   # merah muda pastel
    else:
        return "#7BAEC8"   # biru pastel

all_models["_color"] = all_models.apply(bar_color, axis=1)

fig_u = go.Figure()

fig_u.add_trace(go.Bar(
    x=all_models["Theil's U"],
    y=all_models["Model"],
    orientation="h",
    marker_color=all_models["_color"].tolist(),
    marker_line=dict(color="rgba(0,0,0,0.08)", width=0.8),
    text=[f"{u:.3f}" for u in all_models["Theil's U"]],
    textposition="outside",
    textfont=dict(size=11, color="#4A5568"),
    hovertemplate="<b>%{y}</b><br>Theil's U: %{x:.3f}<extra></extra>",
))

# Garis baseline U=1.0
fig_u.add_vline(
    x=1.0, line_dash="dash", line_color="#C0392B", line_width=1.5,
    annotation_text="Baseline Naive (U=1.0)",
    annotation_position="top right",
    annotation_font=dict(size=10, color="#C0392B"),
)

fig_u.update_layout(
    height=460,
    margin=dict(l=10, r=80, t=20, b=20),
    xaxis=dict(
        title="Theil's U (lebih kecil = lebih baik)",
        range=[0, 1.35],
        gridcolor="#F7FAFC",
        tickfont=dict(size=11, color="#4A5568"),
        title_font=dict(size=12, color="#2D3748"),
        linecolor="#CBD5E0",
    ),
    yaxis=dict(
        autorange="reversed",
        tickfont=dict(size=11, color="#4A5568"),
        linecolor="#CBD5E0",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=11, color="#2D3748"),
)
st.plotly_chart(fig_u, use_container_width=True)

# Legenda warna manual (sederhana)
leg_c1, leg_c2, leg_c3, _ = st.columns([1, 1, 1, 3])
leg_c1.markdown(
    '<span style="display:inline-block;width:14px;height:14px;'
    'background:#B07A8A;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
    '<span style="font-size:.8rem;color:#4A5568;">Terbaik overall</span>',
    unsafe_allow_html=True,
)
leg_c2.markdown(
    '<span style="display:inline-block;width:14px;height:14px;'
    'background:#D4A0A8;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
    '<span style="font-size:.8rem;color:#4A5568;">Siklus 2 Hibrida</span>',
    unsafe_allow_html=True,
)
leg_c3.markdown(
    '<span style="display:inline-block;width:14px;height:14px;'
    'background:#7BAEC8;border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
    '<span style="font-size:.8rem;color:#4A5568;">Siklus 1 Standalone</span>',
    unsafe_allow_html=True,
)

st.divider()

# ===========================================================================
# SECTION 4 -- Tren PDRB Historis (semua provinsi)
# ===========================================================================
st.markdown("#### Tren PDRB Growth Historis per Provinsi (2011-2024)")

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
            x="2020-04-01", line_dash="dot", line_color="#C0392B",
            annotation_text="COVID Q2-2020", annotation_position="top right",
        )
    fig_hist.add_hline(y=0, line_color="#718096", line_width=0.8)
    fig_hist.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=20),
        yaxis_title="PDRB Growth YoY (%)",
        legend=dict(orientation="h", y=-0.2, font=dict(size=10, color="#2D3748")),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=11, color="#2D3748"),
        xaxis=dict(
            gridcolor="#F7FAFC",
            tickfont=dict(size=11, color="#4A5568"),
            title_font=dict(size=12, color="#2D3748"),
            linecolor="#CBD5E0",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#4A5568"),
            title_font=dict(size=12, color="#2D3748"),
            linecolor="#CBD5E0",
        ),
    )
    fig_hist.update_xaxes(gridcolor="#F7FAFC")
    fig_hist.update_yaxes(gridcolor="#F7FAFC")
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ===========================================================================
# SECTION 5 -- Proyeksi ringkasan (jika ada pkl)
# ===========================================================================
st.markdown("#### Proyeksi PDRB Growth 2025-2027 · Naive+XGB_strict")

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