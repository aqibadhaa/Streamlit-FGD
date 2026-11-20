"""
utils/data_loader.py
Loader terpusat — semua data & model di-cache agar tidak re-load tiap interaksi.

Langkah setup (jalankan di luar Streamlit, dari notebook):
    import joblib
    joblib.dump(final_xgb_strict,  'assets/model_xgb_strict.pkl')
    joblib.dump(best_alphas_11,    'assets/best_alphas.pkl')
    joblib.dump(df_proj_out,       'assets/df_proj_out.pkl')
    joblib.dump(shap_values_arr,   'assets/shap_values.pkl')
    joblib.dump(X_shap,            'assets/X_shap.pkl')
    joblib.dump(FEATURE_COLS,      'assets/feature_cols.pkl')
    df.to_csv('assets/dataset_panel_jawa.csv', index=False)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

PROVINSI_LIST = [
    "Banten", "DI Yogyakarta", "DKI Jakarta",
    "Jawa Barat", "Jawa Tengah", "Jawa Timur",
]

FEATURES = ["Inflasi_MtoM_Pct", "TPT_Pct", "BI_Rate", "PMA_JutaUSD"]
TARGET   = "PDRB_Growth"

KUARTAL_MAP = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

# ── Palet warna provinsi (konsisten di semua halaman) ──────────────────────
PROV_COLORS = {
    "Banten":        "#0A6E8A",
    "DI Yogyakarta": "#12B4D2",
    "DKI Jakarta":   "#F59E0B",
    "Jawa Barat":    "#10B981",
    "Jawa Tengah":   "#6366F1",
    "Jawa Timur":    "#EF4444",
}

# ── Loaders (cached) ───────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    path = os.path.join(ASSETS, "dataset_panel_jawa.csv")
    if not os.path.exists(path):
        return _generate_synthetic_data()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model():
    """Return (model_xgb, best_alphas) atau None jika belum ada."""
    path_model  = os.path.join(ASSETS, "model_xgb_strict.pkl")
    path_alphas = os.path.join(ASSETS, "best_alphas.pkl")
    if os.path.exists(path_model) and os.path.exists(path_alphas):
        return joblib.load(path_model), joblib.load(path_alphas)
    return None, None


@st.cache_data(show_spinner=False)
def load_projections() -> pd.DataFrame:
    path = os.path.join(ASSETS, "df_proj_out.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_shap():
    """Return (shap_values_arr, X_shap, feature_cols) atau None."""
    p_sv = os.path.join(ASSETS, "shap_values.pkl")
    p_x  = os.path.join(ASSETS, "X_shap.pkl")
    p_fc = os.path.join(ASSETS, "feature_cols.pkl")
    if all(os.path.exists(p) for p in [p_sv, p_x, p_fc]):
        return joblib.load(p_sv), joblib.load(p_x), joblib.load(p_fc)
    return None, None, None


# ── Synthetic fallback (demo tanpa file pkl) ───────────────────────────────

def _generate_synthetic_data() -> pd.DataFrame:
    """
    Data sintetis untuk demo UI tanpa file asli.
    Ganti dengan CSV asli saat deploy.
    """
    rng = np.random.default_rng(42)
    rows = []
    base_growth = {
        "DKI Jakarta": 5.8, "Jawa Barat": 5.2, "Jawa Tengah": 5.0,
        "Jawa Timur": 5.5, "DI Yogyakarta": 4.8, "Banten": 5.1,
    }
    for prov in PROVINSI_LIST:
        for tahun in range(2011, 2027):
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                covid_shock = -11 if (tahun == 2020 and q == "Q2") else 0
                recovery    =  10 if (tahun == 2021 and q == "Q2") else 0
                growth = (
                    base_growth[prov]
                    + rng.normal(0, 0.6)
                    + covid_shock
                    + recovery
                )
                rows.append({
                    "Provinsi": prov,
                    "Tahun": tahun,
                    "Kuartal": q,
                    "Period": f"{tahun}-{q}",
                    TARGET: round(growth, 4) if tahun <= 2024 else np.nan,
                    "Inflasi_MtoM_Pct": round(rng.uniform(0.1, 0.9), 3),
                    "TPT_Pct": round(rng.uniform(4.5, 9.0), 2),
                    "BI_Rate": round(rng.choice([3.5, 4.0, 4.75, 5.5, 5.75, 6.0, 6.25]), 2),
                    "PMA_JutaUSD": round(rng.uniform(200, 3000), 1),
                })
    return pd.DataFrame(rows)
