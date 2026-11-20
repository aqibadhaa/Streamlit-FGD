# PDRB Jawa Forecast — Streamlit App
**FGD Kelompok 1 · Metodologi CRISP-DM**

## Struktur Aplikasi

```
streamlit_app/
├── app.py                        ← Entry point & landing page
├── pages/
│   ├── 1_🏠_Dashboard.py         ← KPI, leaderboard model, tren historis
│   ├── 2_🔮_Proyeksi_Interaktif.py ← Simulasi skenario & proyeksi 2025–2027
│   └── 3_🧠_Interpretasi_SHAP.py  ← XAI — kontribusi variabel makro
├── utils/
│   └── data_loader.py            ← Loader terpusat (cached)
├── assets/                       ← Dataset CSV + file pkl model
│   ├── dataset_panel_jawa.csv
│   ├── model_xgb_strict.pkl
│   ├── best_alphas.pkl
│   ├── df_proj_out.pkl
│   ├── shap_values.pkl
│   ├── X_shap.pkl
│   └── feature_cols.pkl
└── requirements.txt
```

## Setup: Export dari Notebook

Tambahkan cell berikut di **akhir notebook** (setelah Bab 8):

```python
import joblib, os

os.makedirs("assets", exist_ok=True)

# 1. Dataset CSV
df.to_csv("assets/dataset_panel_jawa.csv", index=False)

# 2. Model terbaik: Naive+XGB_strict
joblib.dump(final_xgb_strict, "assets/model_xgb_strict.pkl")
joblib.dump(best_alphas_11,   "assets/best_alphas.pkl")

# 3. Hasil proyeksi rekursif 2025–2027
joblib.dump(df_proj_out, "assets/df_proj_out.pkl")

# 4. SHAP artifacts (dari Sub-Bab 5.6)
joblib.dump(shap_values_arr, "assets/shap_values.pkl")
joblib.dump(X_shap,          "assets/X_shap.pkl")
joblib.dump(FEATURE_COLS,    "assets/feature_cols.pkl")

print("Export selesai!")
```

> **Catatan**: Tanpa file pkl, app tetap bisa berjalan menggunakan
> data sintetis dan proyeksi ilustratif (untuk demo UI).

## Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Cloud

1. Push seluruh folder `streamlit_app/` ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → pilih `app.py` sebagai entry point
4. Done — URL publik langsung tersedia

## Fitur Interaktif

| Halaman | Interaktivitas |
|---|---|
| Dashboard | Filter provinsi, toggle annotasi COVID |
| Proyeksi | Pilih provinsi, model, horizon, simulasi makro (BI Rate/Inflasi/TPT/PMA), preset skenario, download CSV |
| SHAP | Pilih provinsi, geser observasi untuk waterfall, heatmap lintas provinsi |
