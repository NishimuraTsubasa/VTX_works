# Project Structure

```text
global_equity_futures_evidence_ensemble_v3/
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── run_pipeline.py
├── run_all.sh
├── config/
│   ├── config.yaml
│   └── evidence.yaml
├── data/
│   ├── input/
│   │   ├── Universe_Master.xlsx
│   │   ├── Cross_Asset_Data.xlsx
│   │   └── market/Market_<asset_id>.xlsx
│   └── processed/
│       ├── Monthly_Model_Dataset.xlsx
│       └── Daily_Signal_Sample.xlsx
├── outputs/
│   ├── data/Model_Outputs.xlsx
│   ├── report/Model_Report.pdf
│   └── summary/Summary.xlsx
├── src/
│   ├── excel_io.py
│   ├── data_generation.py
│   ├── features.py
│   ├── models.py
│   ├── ensemble.py
│   ├── portfolio.py
│   ├── reporting.py
│   └── build_excel_summary.py
└── tests/smoke_test.py
```

## Bloomberg preprocessing module (Version 3)

```text
bbg_preprocess/
├─ config/BBG_Config.xlsx
├─ README_BBG_PREPROCESS.md
├─ requirements_bbg.txt
├─ run_bbg_preprocess.py
├─ run_preprocess.sh
├─ logs/
│  ├─ BBG_Query_Log.xlsx
│  └─ BBG_Data_Quality.xlsx
└─ src/bbg_preprocess/
   ├─ config_loader.py
   ├─ provider.py
   ├─ cache.py
   ├─ excel_output.py
   └─ pipeline.py

data/bbg_cache/
└─ Ticker単位の差分取得キャッシュ（Parquet。pyarrow未導入時のMock testはPickle fallback）
```
