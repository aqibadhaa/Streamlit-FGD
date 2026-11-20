# PDRB Jawa Forecast — Streamlit App
**FGD Kelompok 1 · Metodologi CRISP-DM**

## Struktur Aplikasi

```
streamlit_app/
├── app.py                        ← Entry point & landing page
├── pages/
│   ├── Dashboard.py         ← KPI, leaderboard model, tren historis
│   ├── Proyeksi_Interaktif.py ← Simulasi skenario & proyeksi 2025–2027
│   └── Interpretasi_SHAP.py  ← XAI — kontribusi variabel makro
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



## Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```



## Fitur Interaktif

| Halaman | Interaktivitas |
|---|---|
| Dashboard | Filter provinsi, toggle annotasi COVID |
| Proyeksi | Pilih provinsi, model, horizon, simulasi makro (BI Rate/Inflasi/TPT/PMA), preset skenario, download CSV |
| SHAP | Pilih provinsi, geser observasi untuk waterfall, heatmap lintas provinsi |
