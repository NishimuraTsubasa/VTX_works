# 個別銘柄スコアリングモデル比較・評価プロジェクト v0.7

## 1. 目的

本プロジェクトは、個別銘柄のファクター値から作るスコアリングモデルを段階的に改良し、各変更が翌期個別銘柄リターンの順位付け精度へ与える影響を比較するPython実装です。


主な評価対象は次の通りです。

- 個別銘柄スコアと翌期リターンのRankIC
- スコア5分位ポートフォリオの翌期リターン
- Q1-Q5の単調性
- Q5-Q1ロング・ショート累積リターン
- Q5-Q1 Sharpe Ratio・最大ドローダウン
- 単一ファクター候補モデルのOOS評価

## 2. 比較するスコアリングパターン

| ID | パターン | 主な変更点 |
|---|---|---|
| S00 | Current Direct EW | 各FA列を0-1順位化して直接等ウェイト |
| S01 | Missing Adjusted EW | 欠損ファクターを除いてウェイト再正規化 |
| S02 | Winsorized Direct EW | 外れ値処理を追加 |
| S03 | Neutralized Direct EW | 国・セクター・サイズ中立化を追加 |
| S04 | Hierarchical Equal Weight | グループ内・グループ間を等ウェイト |
| S05 | Correlation Adjusted IC | グループ内を相関調整ICウェイト |
| S06 | Selected Factor Models | 原系列・設定済み差分・移動平均乖離を含め、Linear・Piecewise・Quadratic・Combined Ridgeから選択 |
| S07 | Full OOF Ridge | 原系列・派生系列、設定済みグループ統合とOOF Ridgeによる最終候補 |

S00-S05は原系列ファクターだけを使用し、S06-S07でExcel設定済みの差分・移動平均乖離等を追加します。これにより、派生特徴量を導入する増分効果を段階比較できます。

すべてのパターンを同じ5分位評価方法で比較するため、スコア作成方法の差を確認できます。

## 3. フォルダ構成

```text
stock_scoring_model_v7/
├─ README.md
├─ config/
│  └─ model_config.py
├─ data/
│  ├─ input/
│  │  ├─ factors_and_returns.xlsx
│  │  └─ factor_master.xlsx
│  └─ templates/
│     ├─ 01_factors_and_returns_template.xlsx
│     └─ 05_factor_master_template.xlsx
├─ docs/
│  ├─ data_dictionary.md
│  ├─ factor_master_guide.md
│  ├─ file_inventory.md
│  ├─ methodology.md
│  ├─ model_selection.md
│  ├─ derived_factor_features.md
│  └─ output_management.md
├─ outputs/
│  ├─ analysis_summary.xlsx
│  ├─ file_inventory.xlsx
│  ├─ factor_scatter_diagnostics.pdf
│  ├─ factor_bin_diagnostics.pdf
│  ├─ factor_model_selection_report.pdf
│  ├─ factor_model_performance.pdf
│  ├─ quintile_cumulative_returns.pdf
│  ├─ stock_scoring_scenario_comparison.pdf
│  ├─ history/
│  └─ stock_score_patterns/
├─ scripts/
│  ├─ generate_demo_data.py
│  └─ run_pipeline.py
├─ src/stock_scoring_model/
└─ tests/
```

## 4. 必須インプット

### 4.1 factors_and_returns.xlsx

`data` シートへ、時点・銘柄単位のデータを横持ちで入力します。

| 列 | 必須 | 内容 |
|---|---|---|
| date | Yes | スコア計算時点 |
| ISIN | Yes | 銘柄キー |
| stock_return | Yes | 個別銘柄リターン。内部で1期先へシフト |
| market_cap | Yes | 時価総額 |
| currency | Yes | 銘柄通貨 |
| country | Yes | 国 |
| sector | Yes | セクター |
| FA0101等 | Yes | ファクター値 |

### 4.2 factor_master.xlsx

- `Factor_Master`: FAコード、名称、グループ、方向、個別設定
- `Group_Settings`: Value等のグループ統合方法
- `Group_Method_Params`: IC・PCA等の上書き設定
- `Feature_Engineering_Control`: グループ・FAコード単位の派生特徴量生成モード
- `Derived_Feature_Rules`: 差分・移動平均乖離等の生成式、窓、情報ラグ、選択フラグ
- `README`: 入力方法


## 4.3 差分・移動平均乖離ファクター

ValueやMomentum等、Excelで指定したグループまたは個別FAコードについて、原系列から派生特徴量を自動生成できます。

標準設定では、スコア時点 `t` の派生値に `t-1` 以前の情報だけを使います。

```text
1期差分              = x[t-1] - x[t-2]
12期移動平均乖離     = x[t-1] - mean(x[t-2], ..., x[t-13])
```

翌期リターン `r[t+1]` と結合するため、最新使用情報 `x[t-1]` からターゲットまでの間隔は2期です。

`Generation_Mode=all` ならEnabledな派生ルールをすべて使用し、`selected` ならSelected=1だけを使用します。原系列を併用するかは `Include_Raw` で設定します。詳細は `docs/derived_factor_features.md` を参照してください。

## 5. パターン別Excel

`outputs/stock_score_patterns/` に、1パターンにつき1ファイルを出力します。

各ファイルの主要シートは次の通りです。

### StockScore_001

列数を抑え、次の項目だけを横持ちで出力します。

| 列 | 内容 |
|---|---|
| Date | スコア計算時点 |
| ISIN | 銘柄キー |
| Currency | 通貨 |
| MarketCap | 時価総額 |
| TotalScore | 0-1の最終個別銘柄スコア |
| Prediction | 順位化前の予測シグナル |
| NextMonthReturn | 翌期個別銘柄リターン |
| Quintile | 1=最低20%、5=最高20% |

### SubScore_001

Value・Momentum・Quality等を縦持ちで出力します。

```text
Date / ISIN / SubScore / SubScoreValue
```

### FactorScore_001

FA0101等のファクター別スコアを縦持ちで出力します。

```text
Date / ISIN / FactorCode / FactorScore
```

列数を増やさずに多数のファクターを保存できます。行数が設定上限を超えた場合は、同じファイル内で `_001`, `_002` のように分割します。

デフォルトではファイル容量を抑えるため、パターン別Excelは最新時点だけを出力します。全履歴を出す場合はConfigの `date_scope` を `all` に変更します。

## 6. 目視評価用PDF

| ファイル | 内容 |
|---|---|
| `quintile_cumulative_returns.pdf` | 各パターンのQ1-Q5累積リターン |
| `stock_scoring_scenario_comparison.pdf` | Q5-Q1累積、ローリングRankIC、主要指標比較 |
| `factor_scatter_diagnostics.pdf` | ファクター値・翌期リターン散布図と4候補回帰線 |
| `factor_bin_diagnostics.pdf` | 単一ファクター5分位別リターン |
| `factor_model_selection_report.pdf` | 各ファクターのモデル選択根拠 |
| `factor_model_performance.pdf` | 候補モデルのOOS RankIC推移 |

望ましい結果は、Q1からQ5に向かって累積リターンが概ね順序立ち、Q5-Q1累積リターンが安定的に上昇し、RankICが複数期間で正となることです。

## 7. 主な比較指標

- Mean RankIC
- RankIC IR
- RankIC Positive Rate
- Q5-Q1平均リターン
- Q5-Q1年率リターン
- Q5-Q1 Sharpe Ratio
- Q5-Q1最大ドローダウン
- 5分位単調性
- 隣接分位単調性

結果は `analysis_summary.xlsx` の `Scenario_Comparison` に集約されます。

## 8. 実行方法

```bash
cd stock_scoring_model_v7
pip install -r requirements.txt
python scripts/run_pipeline.py --config config/model_config.py
```

または、パッケージをインストールして実行します。

```bash
pip install -e .
stock-scoring-model --config config/model_config.py
```

## 9. 出力制御

### パターン別Excel

```python
"scenario_excel": {
    "date_scope": "latest",  # all / latest / selected
    "include_sub_scores": True,
    "include_factor_scores": True,
    "scenarios": {
        "S00_Current_Direct_EW": True,
        "S07_Full_OOF_Ridge": True,
    },
}
```

### PDF

```python
"pdf": {
    "reports": {
        "scenario_quintile_cumulative": True,
        "scenario_comparison": True,
        "factor_scatter": True,
    }
}
```

### 履歴Excel

履歴は `outputs/history/` へデータ種類ごとに1ファイルずつ出力します。大容量の個別銘柄履歴は初期設定で停止しています。

## 10. 注意事項

- 同梱データと出力は動作確認用の合成データです。
- 実データではPoint-in-Time整合を必ず確認してください。
- S05以降のウェイト・回帰には、予測時点より前の実現リターンだけを使用します。
- Source_Lag=1の派生特徴量はスコア時点tにt-1以前の情報だけを格納し、t+1リターンと評価します。
- パターン比較は共通評価期間のみに限定しています。
- 複雑モデルの改善が小さい場合は、単純モデルを優先します。
