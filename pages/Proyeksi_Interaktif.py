"""
Page 2 — Proyeksi Interaktif
User bisa pilih provinsi, model, dan simulasi skenario makro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import (
    load_dataset, load_projections,
    PROVINSI_LIST, PROV_COLORS, TARGET, FEATURES, KUARTAL_MAP,
)

st.set_page_config(page_title="Proyeksi · PDRB Jawa", page_icon="🔮", layout="wide")

st.markdown("""
<style>
.accent-bar { height:4px; background:linear-gradient(90deg,#0A6E8A,#12B4D2);
              border-radius:2px; margin-bottom:1.2rem; }
.info-box { background:#EFF6FF; border-left:4px solid #3B82F6; padding:.8rem 1rem;
            border-radius:0 8px 8px 0; font-size:.88rem; margin:.6rem 0; }
.warn-box { background:#FEF3C7; border-left:4px solid #F59E0B; padding:.8rem 1rem;
            border-radius:0 8px 8px 0; font-size:.88rem; margin:.6rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.title("🔮 Proyeksi Interaktif")
st.caption("Eksplorasi prediksi PDRB Growth per provinsi dan simulasi skenario makroekonomi")

df      = load_dataset()
df_proj = load_projections()

# ─── Helper: recursive forecast sederhana (tanpa pkl model) ───────────────
def naive_xgb_fallback(prov, df_hist, inflasi, tpt, bi_rate, pma, horizon=12):
    """
    Fallback deterministik saat model pkl belum ada.
    Menggunakan rata-rata growth historis + faktor makro linear sederhana.
    Hanya untuk demo UI; ganti dengan model pkl asli di deployment.
    """
    hist_p  = df_hist[(df_hist["Provinsi"] == prov) & (df_hist["Tahun"] <= 2024)]
    mu      = hist_p[TARGET].dropna().mean()
    sigma   = hist_p[TARGET].dropna().std()

    # Faktor makro sederhana (illustrative)
    bi_effect    = -(bi_rate  - 6.0) * 0.3
    inflasi_eff  = -(inflasi  - 0.3) * 0.5
    tpt_eff      = -(tpt      - 6.5) * 0.2
    pma_eff      =  (pma      - 1500) / 1500 * 0.4
    macro_adj    = bi_effect + inflasi_eff + tpt_eff + pma_eff

    preds = []
    base = mu + macro_adj
    for i in range(horizon):
        # Slight decay toward mu over time
        val = base * 0.85 + mu * 0.15 + np.random.default_rng(42 + i).normal(0, sigma * 0.15)
        preds.append(round(val, 4))
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi Proyeksi")
    st.divider()

    prov = st.selectbox("Pilih Provinsi", PROVINSI_LIST)

    model_choice = st.radio(
        "Model",
        ["Naive+XGB_strict ⭐ (Terbaik)", "Naive+RF_strict", "Ensemble A"],
        index=0,
    )

    horizon_q = st.slider(
        "Horizon Proyeksi (kuartal ke depan)",
        min_value=4, max_value=12, value=8, step=4,
        help="Dihitung dari Q1 2025. Max 12 kuartal = s.d. Q4 2027",
    )

    st.divider()
    st.markdown("### 🔧 Simulasi Skenario Makro")
    st.caption("Geser slider untuk melihat dampak asumsi makroekonomi pada proyeksi")

    bi_rate  = st.slider("BI Rate (%)",              3.0,  10.0, 6.0,  0.25)
    inflasi  = st.slider("Inflasi M-to-M (%)",       -2.0,  5.0, 0.3,  0.1)
    tpt      = st.slider("Tingkat Pengangguran (%)",  3.0,  15.0, 6.5,  0.5)
    pma      = st.slider("PMA (Juta USD)",             0.0, 5000.0, 1500.0, 100.0)

    skenario = st.radio(
        "Preset Skenario",
        ["Custom (manual)", "Optimistik", "Baseline", "Pesimistik"],
        index=2,
    )
    if skenario == "Optimistik":
        bi_rate, inflasi, tpt, pma = 5.25, 0.2, 5.0, 2500.0
        st.success("Skenario Optimistik: BI Rate rendah, TPT rendah, PMA tinggi")
    elif skenario == "Baseline":
        bi_rate, inflasi, tpt, pma = 6.0, 0.3, 6.5, 1500.0
    elif skenario == "Pesimistik":
        bi_rate, inflasi, tpt, pma = 7.5, 1.2, 8.5, 500.0
        st.warning("Skenario Pesimistik: BI Rate tinggi, pengangguran meningkat")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════

# ── Narasi skenario ────────────────────────────────────────────────────────
col_info, col_badge = st.columns([4, 1])
with col_info:
    model_label = model_choice.split(" ")[0] + "+" + model_choice.split("+")[1].split(" ")[0] \
        if "+" in model_choice else model_choice
    st.markdown(f"""
    <div class="info-box">
    <b>Provinsi:</b> {prov} &nbsp;|&nbsp;
    <b>Model:</b> {model_choice} &nbsp;|&nbsp;
    <b>Horizon:</b> {horizon_q} kuartal (s.d. Q{horizon_q%4 or 4} {2024 + (horizon_q//4) + (1 if horizon_q%4 else 0)})<br>
    <b>Asumsi Makro:</b> BI Rate {bi_rate}% · Inflasi {inflasi}% · TPT {tpt}% · PMA USD {pma:,.0f} Juta
    </div>
    """, unsafe_allow_html=True)

# ── Ambil / buat prediksi ──────────────────────────────────────────────────
col_map = {
    "Naive+XGB_strict ⭐ (Terbaik)": "XGB_strict",
    "Naive+RF_strict":               "RF_strict",
    "Ensemble A":                    "Ensemble",
}
col_pred = col_map[model_choice]

quarters = []
for i in range(horizon_q):
    base_q = i % 4   # 0-based kuartal index
    base_y = 2025 + i // 4
    quarters.append((base_y, f"Q{base_q+1}"))

if df_proj is not None and col_pred in df_proj.columns:
    df_p = df_proj[
        (df_proj["Provinsi"] == prov) &
        df_proj.apply(lambda r: (int(r["Tahun"]), r["Kuartal"]) in quarters, axis=1)
    ].sort_values(["Tahun", "Kuartal"])
    preds  = df_p[col_pred].values.tolist()
    actual = df_p["Aktual"].values.tolist()
    has_model = True
else:
    # Fallback deterministik
    preds  = naive_xgb_fallback(prov, df, inflasi, tpt, bi_rate, pma, horizon_q)
    actual = [None] * horizon_q
    has_model = False
    st.markdown("""
    <div class="warn-box">
    ⚠️ File <code>assets/df_proj_out.pkl</code> belum ditemukan — menampilkan proyeksi ilustratif
    berdasarkan rata-rata historis + penyesuaian skenario makro.<br>
    Jalankan notebook hingga Bab 8 lalu export pkl untuk mendapatkan proyeksi model asli.
    </div>
    """, unsafe_allow_html=True)

# ── Susun DataFrame tampilan ───────────────────────────────────────────────
q_labels = [f"{y}-{q}" for y, q in quarters]
df_display = pd.DataFrame({
    "Period": q_labels,
    "Proyeksi": preds,
    "Aktual": actual,
})

# ═══════════════════════════════════════════════════════════════════════════
# CHART — Historis + Proyeksi
# ═══════════════════════════════════════════════════════════════════════════
df_hist = df[(df["Provinsi"] == prov) & (df["Tahun"] <= 2024)].sort_values("Period")

fig = go.Figure()

# Historis aktual
fig.add_trace(go.Scatter(
    x=df_hist["Period"], y=df_hist[TARGET],
    mode="lines+markers",
    name="Aktual 2011–2024",
    line=dict(color=PROV_COLORS[prov], width=2),
    marker=dict(size=3),
))

# Aktual 2025+ (jika ada)
if any(a is not None and not (isinstance(a, float) and np.isnan(a)) for a in actual):
    df_act = df_display[df_display["Aktual"].notna()]
    if len(df_act):
        fig.add_trace(go.Scatter(
            x=df_act["Period"], y=df_act["Aktual"],
            mode="lines+markers",
            name="Aktual 2025+ (realisasi)",
            line=dict(color=PROV_COLORS[prov], width=1.5, dash="dot"),
            marker=dict(size=5, symbol="circle-open"),
        ))

# Proyeksi
fig.add_trace(go.Scatter(
    x=df_display["Period"], y=df_display["Proyeksi"],
    mode="lines+markers",
    name=f"Proyeksi · {model_choice.split(' ')[0]}+{model_choice.split('+')[1].split(' ')[0]}" if "+" in model_choice else f"Proyeksi · {model_choice}",
    line=dict(color="#F59E0B", width=2.2, dash="dash"),
    marker=dict(size=6, symbol="diamond"),
))

# Confidence band sederhana (±0.5 RMSE ilustratif)
rmse_band = 0.38
upper = [p + rmse_band for p in preds]
lower = [p - rmse_band for p in preds]
fig.add_trace(go.Scatter(
    x=q_labels + q_labels[::-1],
    y=upper + lower[::-1],
    fill="toself",
    fillcolor="rgba(245,158,11,0.10)",
    line=dict(color="rgba(0,0,0,0)"),
    showlegend=True,
    name="±RMSE band (0.38%)",
))

# Garis pembatas zona prediksi
fig.add_vline(x="2025-Q1", line_dash="dot", line_color="#6B7280",
              annotation_text="Mulai prediksi", annotation_position="top right")
fig.add_hline(y=0, line_color="#6B7280", line_width=0.6)

fig.update_layout(
    title=dict(text=f"PDRB Growth YoY — {prov}", font=dict(size=14)),
    height=440,
    yaxis_title="PDRB Growth (%)",
    legend=dict(orientation="h", y=-0.22),
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=60),
    font=dict(size=11),
)
fig.update_xaxes(gridcolor="#F1F5F9", tickangle=-45)
fig.update_yaxes(gridcolor="#F1F5F9")

st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TABEL & DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════
col_tbl, col_dl = st.columns([3, 1])

with col_tbl:
    st.markdown(f"#### Tabel Proyeksi · {prov}")
    df_show = df_display.copy()
    df_show["Proyeksi"] = df_show["Proyeksi"].apply(lambda x: f"{x:.4f}%")
    df_show["Aktual"]   = df_show["Aktual"].apply(
        lambda x: f"{x:.4f}%" if (x is not None and not (isinstance(x, float) and np.isnan(x))) else "—"
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

with col_dl:
    st.markdown("#### ⬇️ Unduh")
    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=f"proyeksi_{prov.replace(' ','_')}_{col_pred}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Model: `{col_pred}`  \nSkenario: `{skenario}`  \nProvinsi: `{prov}`")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# SECTION — Semua provinsi sekaligus (jika pkl ada)
# ═══════════════════════════════════════════════════════════════════════════
if df_proj is not None:
    with st.expander("📊 Proyeksi Semua Provinsi Sekaligus", expanded=False):
        fig_all = go.Figure()
        for p in PROVINSI_LIST:
            df_p2 = df_proj[
                (df_proj["Provinsi"] == p) &
                df_proj.apply(lambda r: (int(r["Tahun"]), r["Kuartal"]) in quarters, axis=1)
            ].sort_values(["Tahun", "Kuartal"])
            if len(df_p2) == 0:
                continue
            fig_all.add_trace(go.Scatter(
                x=[f"{int(r.Tahun)}-{r.Kuartal}" for _, r in df_p2.iterrows()],
                y=df_p2[col_pred].values,
                mode="lines+markers",
                name=p,
                line=dict(color=PROV_COLORS[p], width=1.8),
                marker=dict(size=4),
            ))
        fig_all.add_hline(y=0, line_color="#6B7280", line_width=0.6)
        fig_all.update_layout(
            height=380, yaxis_title="PDRB Growth (%)",
            legend=dict(orientation="h", y=-0.25),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=60),
        )
        fig_all.update_xaxes(gridcolor="#F1F5F9", tickangle=-45)
        fig_all.update_yaxes(gridcolor="#F1F5F9")
        st.plotly_chart(fig_all, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION — Insight Otomatis
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("#### 💡 Insight Otomatis")

valid_preds = [p for p in preds if p is not None]
if valid_preds:
    max_val = max(valid_preds)
    min_val = min(valid_preds)
    avg_val = np.mean(valid_preds)
    max_q   = q_labels[preds.index(max_val)]
    min_q   = q_labels[preds.index(min_val)]
    trend   = "meningkat" if preds[-1] > preds[0] else "menurun"

    # BI Rate insight
    if bi_rate > 7.0:
        bi_note = f"BI Rate yang tinggi ({bi_rate}%) berpotensi menekan pertumbuhan melalui efek kontraksi kredit."
    elif bi_rate < 5.0:
        bi_note = f"BI Rate yang rendah ({bi_rate}%) mendukung ekspansi kredit dan mendorong investasi."
    else:
        bi_note = f"BI Rate {bi_rate}% berada pada level netral-moderasi."

    col_i1, col_i2, col_i3 = st.columns(3)
    col_i1.metric("Proyeksi Puncak", f"{max_val:.2f}%", delta=max_q)
    col_i2.metric("Proyeksi Terendah", f"{min_val:.2f}%", delta=min_q)
    col_i3.metric("Rata-rata Proyeksi", f"{avg_val:.2f}%", delta=f"Tren: {trend}")

    st.info(
        f"**{prov}** diprediksi tumbuh rata-rata **{avg_val:.2f}%** YoY selama {horizon_q} kuartal ke depan "
        f"(Q1 2025 s.d. {q_labels[-1]}). Pertumbuhan tertinggi diperkirakan pada **{max_q}** ({max_val:.2f}%). "
        f"{bi_note}"
    )
