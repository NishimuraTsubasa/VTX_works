# ファイル一覧

## 入力

| ファイル | 内容 |
|---|---|
| `data/input/factors_and_returns.xlsx` | 時点 x ISINの属性、当月/翌期リターン、FA値 |
| `data/input/factor_master.xlsx` | FAグループ・方向・統合方法・派生特徴量ルール |
| `config/model_config.py` | 分析頻度、8シナリオ、binscatter、出力制御 |

## 実行コード

| ファイル | 内容 |
|---|---|
| `scripts/run_binscatter.py` | Time-averaged binscatter実行 |
| `scripts/run_pipeline.py` | 8スコアシナリオ・5分位評価実行 |
| `scripts/generate_demo_data.py` | サンプル入力生成 |
| `src/stock_scoring_model/feature_engineering.py` | 差分・移動平均乖離等 |
| `src/stock_scoring_model/preprocessing.py` | Winsorize・中立化・順位化 |
| `src/stock_scoring_model/scenarios.py` | S00-S07のスコア構築 |
| `src/stock_scoring_model/evaluation.py` | RankIC・5分位評価 |
| `src/stock_scoring_model/binscatter.py` | ビン化・time-average・回帰・PDF |
| `src/stock_scoring_model/reporting.py` | PDF・Excel出力 |

## サンプルPDF

| ファイル | 内容 |
|---|---|
| `outputs/diagnostics/binscatter_all_universe.pdf` | 全銘柄FA有効性 |
| `outputs/diagnostics/binscatter_by_country.pdf` | 国別FA有効性 |
| `outputs/diagnostics/binscatter_by_country_sector.pdf` | 国 x セクターFA有効性 |
| `outputs/quintile_cumulative_returns.pdf` | 8シナリオ別Q1-Q5累積リターン |
| `outputs/stock_scoring_scenario_comparison.pdf` | 8シナリオのQ5-Q1、RankIC等の横比較 |

## Excel出力

| ファイル | 内容 |
|---|---|
| `outputs/diagnostics/binscatter_regression_summary.xlsx` | R2、相関、係数、ビン座標 |
| `outputs/analysis_summary.xlsx` | 8シナリオ比較、分位、RankIC、派生系譜 |
| `outputs/stock_score_patterns/S00_Current_Direct_EW.xlsx` | 現行直接等ウェイト |
| `outputs/stock_score_patterns/S01_Missing_Adjusted_EW.xlsx` | 欠損ウェイト再調整 |
| `outputs/stock_score_patterns/S02_Winsorized_Direct_EW.xlsx` | Winsorize追加 |
| `outputs/stock_score_patterns/S03_Neutralized_Direct_EW.xlsx` | 中立化追加 |
| `outputs/stock_score_patterns/S04_Hierarchical_Equal_Weight.xlsx` | 階層等ウェイト |
| `outputs/stock_score_patterns/S05_Correlation_Adjusted_IC.xlsx` | 相関調整IC |
| `outputs/stock_score_patterns/S06_Selected_Factor_Models.xlsx` | 4候補モデル選択 |
| `outputs/stock_score_patterns/S07_Full_OOF_Ridge.xlsx` | OOF Ridge最終統合 |

パターン別Excelは、列数を抑えるため次の階層に分けます。

- `StockScore_001`: Date、ISIN、Currency、MarketCap、Prediction、NextMonthReturn、TotalScore、Quintile
- `SubScore_001`: Date、ISIN、SubScore、SubScoreValue
- `FactorScore_001`: Date、ISIN、FactorCode、FactorScore
- `WeightHistory_001`: S05のFAウェイト等
- `ModelSelection_001`: S06/S07のモデル選択結果
