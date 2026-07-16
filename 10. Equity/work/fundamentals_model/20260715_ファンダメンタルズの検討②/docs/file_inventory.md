# ファイル一覧

## 入力

| ファイル | 必須 | 内容 |
|---|---:|---|
| `data/input/factors_and_returns.xlsx` | Yes | 個別銘柄ファクター値、リターン、時価総額、通貨、国、セクター |
| `data/input/factor_master.xlsx` | Yes | ファクター名称、グループ、方向、統合方法 |
| `config/model_config.py` | Yes | 前処理、モデル、評価、出力可否 |

## テンプレート

| ファイル | 内容 |
|---|---|
| `data/templates/01_factors_and_returns_template.xlsx` | 個別銘柄データ入力テンプレート |
| `data/templates/05_factor_master_template.xlsx` | ファクターマスタ入力テンプレート |

## 主要出力

| ファイル | 内容 |
|---|---|
| `outputs/analysis_summary.xlsx` | パターン比較、モデル選択、最新結果、品質・設定 |
| `outputs/file_inventory.xlsx` | 入力・設定・出力一覧 |
| `outputs/quintile_cumulative_returns.pdf` | パターン別Q1-Q5累積リターン |
| `outputs/stock_scoring_scenario_comparison.pdf` | Q5-Q1・RankIC・主要指標の横比較 |
| `outputs/factor_scatter_diagnostics.pdf` | 単一ファクター散布図と回帰線 |
| `outputs/factor_bin_diagnostics.pdf` | 単一ファクター分位分析 |
| `outputs/factor_model_selection_report.pdf` | 単一ファクターモデル採用理由 |
| `outputs/factor_model_performance.pdf` | 単一ファクターモデル精度推移 |

## パターン別Excel

`outputs/stock_score_patterns/S00_...xlsx` から `S07_...xlsx` まで、1パターンにつき1ファイルです。

主要データは次の3種類です。

- `StockScore_NNN`: TotalScore、Prediction、NextMonthReturn、MarketCap、Currency
- `SubScore_NNN`: 銘柄・グループ別SubScore
- `FactorScore_NNN`: 銘柄・FAコード別FactorScore

## 履歴Excel

`outputs/history/` に以下を種類別に出力します。

- `scenario_rank_ic_history.xlsx`
- `scenario_quintile_return_history.xlsx`
- `scenario_long_short_history.xlsx`
- `factor_performance.xlsx`
- `factor_coefficients.xlsx`
- `factor_ic_history.xlsx`
- `group_weight_history.xlsx`
- `pca_loading_history.xlsx`
- `composite_coefficients.xlsx`

## v0.7で追加した設定

`data/input/factor_master.xlsx` に次のシートを追加しています。

- `Feature_Engineering_Control`: グループ・FAコードごとの生成ON/OFF、all/selected、原系列利用
- `Derived_Feature_Rules`: 差分・移動平均乖離等の式、窓、情報ラグ、選択フラグ

`outputs/analysis_summary.xlsx` には次のシートを追加します。

- `Feature_Lineage`
- `Feature_Engineering_Control`
- `Derived_Feature_Rules`
