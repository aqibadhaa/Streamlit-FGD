"""
Prediksi Pertumbuhan Ekonomi Pulau Jawa
CRISP-DM | FGD Kelompok 1
Entry point — Streamlit Multi-page App
"""

import streamlit as st

st.set_page_config(
    page_title="PDRB Jawa Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom global CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar nav label */
    [data-testid="stSidebarNav"] { font-size: 0.9rem; }

    /* Metric card value size */
    [data-testid="stMetricValue"] { font-size: 2rem !important; }

    /* Thin top accent bar */
    .accent-bar {
        height: 4px;
        background: linear-gradient(90deg, #0A6E8A 0%, #12B4D2 100%);
        border-radius: 2px;
        margin-bottom: 1.2rem;
    }

    /* Section header */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #6B7280;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    /* Pill badge */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .badge-green  { background:#D1FAE5; color:#065F46; }
    .badge-blue   { background:#DBEAFE; color:#1E40AF; }
    .badge-orange { background:#FEF3C7; color:#92400E; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar branding ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 PDRB Jawa Forecast")
    st.caption("FGD Kelompok 1 · CRISP-DM")
    st.divider()
    st.markdown("""
    **Navigasi:**
    - 🏠 Dashboard
    - 🔮 Proyeksi Interaktif
    - 🧠 Interpretasi SHAP
    """)
    st.divider()
    st.caption("Dataset: Panel Makroekonomi Kuartalan  \n6 Provinsi Pulau Jawa | 2011–2027")

# ── Landing page ───────────────────────────────────────────────────────────
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)
st.title("Prediksi Pertumbuhan Ekonomi Pulau Jawa")
st.markdown("""
**Metodologi CRISP-DM** · Model Hibrida Naive+XGBoost (Theil's U = 0.880)  
Gunakan menu di sidebar kiri untuk mulai eksplorasi.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**🏠 Dashboard**\nRingkasan KPI, perbandingan model, dan peta pertumbuhan provinsi.")
with col2:
    st.info("**🔮 Proyeksi Interaktif**\nSimulasi skenario makro dan proyeksi PDRB per provinsi 2025–2027.")
with col3:
    st.info("**🧠 Interpretasi SHAP**\nKontribusi tiap variabel makroekonomi terhadap prediksi model.")
