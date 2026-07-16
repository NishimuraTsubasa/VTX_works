# 個別銘柄スコアリングモデル v0.10

個別銘柄ファクターを用いて、現行モデルから新モデルまでを**8つの分析パターンに分解して比較**するPythonプロジェクトです。

本プロジェクトは個別銘柄スコアリングモデルのみを対象とし、指数・指数先物の評価は含みません。

## 1. 実施する分析

- 現行スコア（FA値を0-1順位化して直接等ウェイト）の再現
- 欠損処理、外れ値処理、中立化、階層化の段階比較
- 相関調整ICによるグループ内ウェイト
- Linear / Piecewise / Quadratic / Combined Ridgeの単一ファクターモデル選択
- グループ予測のOOF Ridge統合
- ファクター差分・移動平均乖離などの派生特徴量生成
- スコア5分位ポートフォリオの翌期リターン評価
- RankIC、Q5-Q1、単調性、Sharpe、最大ドローダウンの比較
- Time-averaged binscatterによる単一ファクター有効性の可視化
- 全銘柄、国別、国 x セクター別のLinear / Quadratic / Broken-stick回帰
- 各回帰の決定係数 R2、相関、Top-Bottom差の出力

## 2. 8つの分析パターン

| ID | パターン | 主な変更点 |
|---|---|---|
| S00 | `S00_Current_Direct_EW` | 現行モデル。0-1順位化後、欠損を中立値0.5で補完し全FAを直接等ウェイト |
| S01 | `S01_Missing_Adjusted_EW` | 欠損FAを除外し、銘柄ごとに利用可能FAのウェイトを再正規化 |
| S02 | `S02_Winsorized_Direct_EW` | 1%-99% Winsorizeを追加 |
| S03 | `S03_Neutralized_Direct_EW` | 国・セクター・対数時価総額中立化を追加 |
| S04 | `S04_Hierarchical_Equal_Weight` | グループ内等ウェイト、グループ間等ウェイトへ変更 |
| S05 | `S05_Correlation_Adjusted_IC` | グループ内を過去RankICとFA相関で調整 |
| S06 | `S06_Selected_Factor_Models` | 各元・派生FAについて4候補モデルをOOSで比較し選択 |
| S07 | `S07_Full_OOF_Ridge` | S06のグループ予測をOOF Ridgeで最終統合 |

すべてのパターンで、5分位評価方法を固定します。したがって、各段階でスコア作成ロジックを変更した効果を比較できます。

## 3. フォルダ構成

```text
stock_scoring_model_v10/
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ config/
│  └─ model_config.py
├─ data/
│  └─ input/
│     ├─ factors_and_returns.xlsx
│     └─ factor_master.xlsx
├─ docs/
│  ├─ input_generation_readme.md
│  ├─ config_guide.md
│  ├─ model_logic_and_analysis_flow.md
│  ├─ binscatter_time_averaged_spec.md
│  └─ file_inventory.md
├─ scripts/
│  ├─ run_binscatter.py
│  ├─ run_pipeline.py
│  └─ generate_demo_data.py
├─ src/stock_scoring_model/
│  ├─ feature_engineering.py
│  ├─ preprocessing.py
│  ├─ scenarios.py
│  ├─ evaluation.py
│  ├─ binscatter.py
│  ├─ binscatter_runner.py
│  ├─ reporting.py
│  └─ pipeline.py
├─ tests/
└─ outputs/
```

## 4. インストール

```bash
cd stock_scoring_model_v10
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
source .venv/bin/activate
```

## 5. 実行手順

### Step 1: Time-averaged binscatter

```bash
python scripts/run_binscatter.py --config config/model_config.py
```

### Step 2: 8パターンのスコアリング・5分位評価

```bash
python scripts/run_pipeline.py --config config/model_config.py
```

## 6. 必須入力

### `factors_and_returns.xlsx`

1行は「1時点 x 1銘柄」です。

```text
date
ISIN
stock_return
market_cap
currency
country
sector
FA0101, FA0102, ...
```

詳細は [`docs/input_generation_readme.md`](docs/input_generation_readme.md) を参照してください。

### `factor_master.xlsx`

FAコードのグループ、使用可否、方向、派生特徴量ルールを管理します。`Factor_Name_JP`、`Factor_Name_EN` は使用しません。

## 7. サンプル出力

### ファクター有効性

```text
outputs/diagnostics/
├─ binscatter_all_universe.pdf
├─ binscatter_by_country.pdf
├─ binscatter_by_country_sector.pdf
└─ binscatter_regression_summary.xlsx
```

### 8パターンの個別銘柄スコア評価

```text
outputs/
├─ quintile_cumulative_returns.pdf
├─ stock_scoring_scenario_comparison.pdf
├─ analysis_summary.xlsx
└─ stock_score_patterns/
   ├─ S00_Current_Direct_EW.xlsx
   ├─ S01_Missing_Adjusted_EW.xlsx
   ├─ S02_Winsorized_Direct_EW.xlsx
   ├─ S03_Neutralized_Direct_EW.xlsx
   ├─ S04_Hierarchical_Equal_Weight.xlsx
   ├─ S05_Correlation_Adjusted_IC.xlsx
   ├─ S06_Selected_Factor_Models.xlsx
   └─ S07_Full_OOF_Ridge.xlsx
```

## 8. パターン別Excelの主要項目

- `StockScore_001`: Date、ISIN、Currency、MarketCap、Prediction、NextMonthReturn、TotalScore、Quintile
- `SubScore_001`: Date、ISIN、SubScore、SubScoreValue
- `FactorScore_001`: Date、ISIN、FactorCode、FactorScore
- `WeightHistory_001`: S05等で使用したFAウェイト
- `ModelSelection_001`: S06・S07で採用した候補モデルと選択理由

Configの`date_scope="latest"`では、FactorScore/SubScoreは最新時点だけを出力します。評価用のスコア・リターン履歴は内部で全期間保持します。

## 9. 重要な時点管理

入力の`stock_return`が当月リターンなら、時点`t`の元ファクターは`t+1`リターンへ対応させます。

派生特徴量は`Source_Lag_Periods=1`の場合、`t-1`以前のFA値から作り、`t+1`リターンを評価します。したがって、最新の情報源から目的変数までの実効ギャップは2期間です。

## 10. サンプルデータ

同梱サンプルは動作確認時間を抑えるため、240銘柄・30か月で作成しています。実装ロジックは2,500銘柄以上の月次ユニバースを前提に、出力列を縦持ち・最新時点中心に抑えています。
