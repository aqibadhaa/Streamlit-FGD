"""
Page 2 — Proyeksi Interaktif (v2)
- Pilih provinsi
- Bagian 1: detail per model (5 model terbaik), chart zoomable + tabel
- Bagian 2: komparasi 5 model sekaligus, zona proyeksi murni 2026 Q3 – 2027 Q4
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import (
    load_dataset, load_projections,
    PROVINSI_LIST, PROV_COLORS, TARGET,
)

st.set_page_config(page_title="Proyeksi · PDRB Jawa", page_icon="🔮", layout="wide")

st.markdown("""
<style>
.accent-bar { height:4px; background:linear-gradient(90deg,#0A6E8A,#12B4D2);
              border-radius:2px; margin-bottom:1.2rem; }
.model-card { border-radius:10px; padding:1rem 1.2rem; margin-bottom:.4rem; }
.metric-row { display:flex; gap:2rem; margin-top:.5rem; }
.metric-item { text-align:center; }
.metric-val  { font-size:1.4rem; font-weight:700; color:#0F172A; }
.metric-lbl  { font-size:.7rem; color:#6B7280; text-transform:uppercase;
               letter-spacing:.05em; }
.zone-label  { font-size:.75rem; font-weight:600; color:#6B7280;
               text-transform:uppercase; letter-spacing:.06em; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.title("🔮 Proyeksi Interaktif")
st.caption("Eksplorasi proyeksi murni 2026 Q3 – 2027 Q4 per model dan per provinsi")

# ── Load data ──────────────────────────────────────────────────────────────
df      = load_dataset()
df_proj = load_projections()

# ── 5 Model terbaik ────────────────────────────────────────────────────────
MODELS = {
    "Naive+XGB_strict":  {"col": "XGB_strict",  "rmse": 0.3797, "mape": 6.11, "u": 0.880, "color": "#0A6E8A",  "rank": 1},
    "Ensemble A":        {"col": "Ensemble",     "rmse": 0.3820, "mape": 6.17, "u": 0.885, "color": "#10B981",  "rank": 2},
    "Ensemble B":        {"col": "Ensemble_C",   "rmse": 0.3820, "mape": 6.17, "u": 0.885, "color": "#6366F1",  "rank": 3},
    "Naive+RF_strict":   {"col": "RF_strict",    "rmse": 0.3851, "mape": 6.24, "u": 0.892, "color": "#F59E0B",  "rank": 4},
    "Ensemble C":        {"col": "Ensemble_C",   "rmse": 0.3930, "mape": 6.47, "u": 0.911, "color": "#EF4444",  "rank": 5},
}

# Zona waktu
BACKTEST_START = "2025-Q1"
PURE_PROJ_START = "2026-Q3"   # proyeksi murni mulai sini
PROJ_END        = "2027-Q4"

def period_sort_key(p):
    y, q = p.split("-")
    return int(y) * 10 + int(q[1])

def is_pure_proj(period):
    return period_sort_key(period) >= period_sort_key(PURE_PROJ_START)

def is_backtest(period):
    return period_sort_key(BACKTEST_START) <= period_sort_key(period) < period_sort_key(PURE_PROJ_START)

# ═══════════════════════════════════════════════════════════════════════════
# FILTER PROVINSI
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("### 📍 Pilih Provinsi")
prov = st.selectbox("", PROVINSI_LIST, label_visibility="collapsed")
st.divider()

# ── Siapkan data historis provinsi ────────────────────────────────────────
df_hist = df[(df["Provinsi"] == prov) & (df["Tahun"] <= 2024)].sort_values("Period")

# ── Siapkan data proyeksi provinsi ────────────────────────────────────────
if df_proj is not None:
    df_prov_proj = df_proj[df_proj["Provinsi"] == prov].copy()
    df_prov_proj["Period"] = df_prov_proj["Tahun"].astype(str) + "-" + df_prov_proj["Kuartal"]
    df_prov_proj = df_prov_proj.sort_values("Period")
    # Subset zona murni
    df_pure = df_prov_proj[df_prov_proj["Period"].apply(is_pure_proj)].copy()
    # Subset backtesting (ada aktualnya)
    df_back = df_prov_proj[df_prov_proj["Period"].apply(is_backtest)].copy()
    has_proj = True
else:
    has_proj = False
    st.warning("File `assets/df_proj_out.pkl` belum ditemukan. Pastikan sudah export dari notebook.", icon="⚠️")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 1 — DETAIL PER MODEL
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Bagian 1 — Detail Per Model")
st.caption("Pilih model untuk melihat performa dan proyeksi detailnya")

# ── Pilih model (radio dengan layout horizontal) ───────────────────────────
model_labels = []
for name, info in MODELS.items():
    prefix = "⭐" if info["rank"] == 1 else f"#{info['rank']}"
    model_labels.append(f"{prefix} {name}")

selected_label = st.radio(
    "Model:",
    model_labels,
    horizontal=True,
    index=0,
)
selected_name = selected_label.split(" ", 1)[1]   # strip emoji+rank prefix
sel = MODELS[selected_name]

# ── Info card performa model ───────────────────────────────────────────────
col_badge, col_metrics = st.columns([1, 4])

with col_badge:
    rank_emoji = "🥇" if sel["rank"] == 1 else ("🥈" if sel["rank"] == 2 else ("🥉" if sel["rank"] == 3 else f"#{sel['rank']}"))
    st.markdown(f"""
    <div class="model-card" style="background:{sel['color']}18;
         border:2px solid {sel['color']};">
      <div style="font-size:.8rem;color:#6B7280;">Rank</div>
      <div style="font-size:2rem;font-weight:800;color:{sel['color']};">{rank_emoji}</div>
      <div style="font-size:.85rem;font-weight:700;color:#0F172A;margin-top:.3rem;">{selected_name}</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("RMSE",       f"{sel['rmse']:.4f}", help="Root Mean Squared Error pada test set")
    mc2.metric("MAPE",       f"{sel['mape']:.2f}%", help="Mean Absolute Percentage Error")
    mc3.metric("Theil's U",  f"{sel['u']:.3f}",   help="< 1.0 berarti lebih baik dari Naive baseline")
    mc4.metric("vs Baseline","Lebih Baik ✓" if sel['u'] < 1.0 else "Di bawah baseline",
               delta=f"U={sel['u']:.3f} < 1.0" if sel['u'] < 1.0 else None)

# ── Chart detail model — full timeline + zoom ──────────────────────────────
st.markdown(f"#### Proyeksi · {selected_name} · {prov}")
st.caption("💡 Gunakan scroll mouse atau drag untuk zoom · Double-click untuk reset · Klik legenda untuk hide/show")

col_pred = sel["col"]

fig_detail = go.Figure()

# 1. Historis aktual (2011–2024)
fig_detail.add_trace(go.Scatter(
    x=df_hist["Period"], y=df_hist[TARGET],
    mode="lines",
    name="Aktual 2011–2024",
    line=dict(color="#94A3B8", width=1.8),
    hovertemplate="%{x}: <b>%{y:.2f}%</b><extra>Aktual</extra>",
))

# 2. Backtesting (2025 Q1 – 2026 Q2) — aktual vs prediksi
if len(df_back) > 0 and "Aktual" in df_back.columns:
    df_back_act = df_back[df_back["Aktual"].notna()]
    if len(df_back_act) > 0:
        fig_detail.add_trace(go.Scatter(
            x=df_back_act["Period"], y=df_back_act["Aktual"],
            mode="lines+markers",
            name="Aktual 2025–2026 Q2",
            line=dict(color="#475569", width=2, dash="dot"),
            marker=dict(size=5, symbol="circle"),
            hovertemplate="%{x}: <b>%{y:.2f}%</b><extra>Aktual Realisasi</extra>",
        ))

    if col_pred in df_back.columns:
        fig_detail.add_trace(go.Scatter(
            x=df_back["Period"], y=df_back[col_pred],
            mode="lines+markers",
            name=f"Prediksi Backtesting",
            line=dict(color=sel["color"], width=1.8, dash="dash"),
            marker=dict(size=4, symbol="circle-open"),
            hovertemplate="%{x}: <b>%{y:.2f}%</b><extra>Prediksi Backtest</extra>",
        ))

# 3. Proyeksi murni (2026 Q3 – 2027 Q4) ⭐
if len(df_pure) > 0 and col_pred in df_pure.columns:
    # Sambungkan dari titik terakhir backtesting
    connect_period = []
    connect_val    = []
    if len(df_back) > 0 and col_pred in df_back.columns:
        last_bt = df_back.sort_values("Period").iloc[-1]
        connect_period = [last_bt["Period"]]
        connect_val    = [last_bt[col_pred]]

    all_periods = connect_period + df_pure["Period"].tolist()
    all_vals    = connect_val    + df_pure[col_pred].tolist()

    fig_detail.add_trace(go.Scatter(
        x=all_periods, y=all_vals,
        mode="lines+markers",
        name=f"Proyeksi Murni 2026 Q3–2027 Q4",
        line=dict(color=sel["color"], width=2.8),
        marker=dict(size=7, symbol="diamond", color=sel["color"],
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: <b>%{y:.2f}%</b><extra>Proyeksi Murni</extra>",
    ))

    # Confidence band ±RMSE
    upper = [v + sel["rmse"] for v in all_vals]
    lower = [v - sel["rmse"] for v in all_vals]
    fig_detail.add_trace(go.Scatter(
        x=all_periods + all_periods[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor=f"rgba({int(sel['color'][1:3],16)},{int(sel['color'][3:5],16)},{int(sel['color'][5:7],16)},0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±RMSE ({sel['rmse']:.4f})",
        hoverinfo="skip",
    ))

# Garis pembatas zona
fig_detail.add_vrect(
    x0=PURE_PROJ_START, x1=PROJ_END,
    fillcolor="rgba(245,158,11,0.06)",
    layer="below", line_width=0,
)
fig_detail.add_vline(
    x=PURE_PROJ_START,
    line_dash="dash", line_color="#F59E0B", line_width=1.5,
    annotation_text="▶ Proyeksi Murni",
    annotation_position="top right",
    annotation_font=dict(color="#F59E0B", size=11),
)
fig_detail.add_vline(
    x=BACKTEST_START,
    line_dash="dot", line_color="#94A3B8", line_width=1.2,
    annotation_text="▶ Backtesting",
    annotation_position="top right",
    annotation_font=dict(color="#94A3B8", size=10),
)
fig_detail.add_hline(y=0, line_color="#CBD5E1", line_width=0.8)

fig_detail.update_layout(
    height=480,
    yaxis_title="PDRB Growth YoY (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.22, x=0),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=10, t=20, b=80),
    font=dict(size=11),
    # Aktifkan rangeslider untuk zoom mudah
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.06),
        type="category",
        gridcolor="#F1F5F9",
    ),
)
fig_detail.update_yaxes(gridcolor="#F1F5F9")
st.plotly_chart(fig_detail, use_container_width=True)

# ── Tabel proyeksi murni ───────────────────────────────────────────────────
if len(df_pure) > 0 and col_pred in df_pure.columns:
    st.markdown(f"##### Tabel Proyeksi Murni · {selected_name} · {prov}")

    tbl_cols = ["Period", col_pred]
    if "Aktual" in df_pure.columns:
        tbl_cols = ["Period", "Aktual", col_pred]

    df_tbl = df_pure[tbl_cols].copy()
    df_tbl = df_tbl.rename(columns={
        col_pred: f"Proyeksi ({selected_name})",
        "Aktual": "Aktual (jika tersedia)",
    })

    # Format
    for c in df_tbl.columns:
        if c != "Period":
            df_tbl[c] = df_tbl[c].apply(
                lambda x: f"{x:.4f}%" if pd.notna(x) else "—"
            )

    col_tbl, col_dl = st.columns([3, 1])
    with col_tbl:
        st.dataframe(df_tbl, use_container_width=True, hide_index=True)
    with col_dl:
        csv = df_pure[[col_pred]].assign(Period=df_pure["Period"]).to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download CSV",
            data=csv,
            file_name=f"proyeksi_murni_{prov.replace(' ','_')}_{selected_name.replace('+','_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# BAGIAN 2 — KOMPARASI 5 MODEL SEKALIGUS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Bagian 2 — Komparasi Semua Model")
st.caption(f"5 model terbaik ditampilkan bersamaan · Zona proyeksi murni 2026 Q3 – 2027 Q4 · {prov}")

fig_comp = go.Figure()

# Latar zona proyeksi murni
fig_comp.add_vrect(
    x0=PURE_PROJ_START, x1=PROJ_END,
    fillcolor="rgba(245,158,11,0.07)",
    layer="below", line_width=0,
    annotation_text="PROYEKSI MURNI",
    annotation_position="top left",
    annotation_font=dict(color="#B45309", size=10, family="monospace"),
)

# Garis nol
fig_comp.add_hline(y=0, line_color="#CBD5E1", line_width=0.8)

# Garis pembatas zona murni
fig_comp.add_vline(
    x=PURE_PROJ_START,
    line_dash="dash", line_color="#F59E0B", line_width=1.5,
)

# Plot semua 5 model — hanya zona proyeksi (backtest + murni) agar chart fokus
df_proj_zone = df_prov_proj[
    df_prov_proj["Period"].apply(lambda p: period_sort_key(p) >= period_sort_key(BACKTEST_START))
].copy()

for name, info in MODELS.items():
    col = info["col"]
    if col not in df_proj_zone.columns:
        continue

    is_best = info["rank"] == 1
    fig_comp.add_trace(go.Scatter(
        x=df_proj_zone["Period"],
        y=df_proj_zone[col],
        mode="lines+markers",
        name=f"{'⭐ ' if is_best else ''}{name}  (U={info['u']:.3f})",
        line=dict(
            color=info["color"],
            width=3.0 if is_best else 1.8,
            dash="solid" if is_best else "solid",
        ),
        marker=dict(
            size=8 if is_best else 5,
            symbol="diamond" if is_best else "circle",
        ),
        hovertemplate=f"%{{x}}: <b>%{{y:.2f}}%</b><extra>{name}</extra>",
        opacity=1.0 if is_best else 0.75,
    ))

# Aktual backtesting sebagai referensi (abu tipis)
if "Aktual" in df_proj_zone.columns:
    df_act_ref = df_proj_zone[df_proj_zone["Aktual"].notna()]
    if len(df_act_ref) > 0:
        fig_comp.add_trace(go.Scatter(
            x=df_act_ref["Period"], y=df_act_ref["Aktual"],
            mode="lines+markers",
            name="Aktual Realisasi",
            line=dict(color="#1E293B", width=2, dash="dot"),
            marker=dict(size=5, color="#1E293B"),
            hovertemplate="%{x}: <b>%{y:.2f}%</b><extra>Aktual</extra>",
        ))

fig_comp.update_layout(
    height=440,
    yaxis_title="PDRB Growth YoY (%)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=10, t=20, b=90),
    font=dict(size=11),
    xaxis=dict(gridcolor="#F1F5F9", tickangle=-45),
)
fig_comp.update_yaxes(gridcolor="#F1F5F9")
st.plotly_chart(fig_comp, use_container_width=True)

# ── Tabel ringkas komparasi akhir proyeksi ─────────────────────────────────
st.markdown(f"##### Ringkasan Nilai Akhir Proyeksi (2027 Q4) · {prov}")

last_period = df_pure["Period"].max() if len(df_pure) > 0 else PROJ_END
df_last = df_prov_proj[df_prov_proj["Period"] == last_period]

rows = []
for name, info in MODELS.items():
    col = info["col"]
    val = df_last[col].values[0] if (len(df_last) > 0 and col in df_last.columns) else None
    rows.append({
        "Rank": f"{'⭐' if info['rank']==1 else info['rank']}",
        "Model": name,
        "RMSE": f"{info['rmse']:.4f}",
        "MAPE": f"{info['mape']:.2f}%",
        "Theil's U": f"{info['u']:.3f}",
        f"Proyeksi {last_period} (%)": f"{val:.4f}" if val is not None else "—",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
