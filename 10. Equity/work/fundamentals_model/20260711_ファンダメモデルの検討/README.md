# 株式ファクターから国別株価指数先物へ接続するスコアリングモデル v0.4

## 1. 目的

本プロジェクトは、個別銘柄のファクター値と翌期リターンを用いてファクターの有効性を検証し、個別銘柄スコアを作成した後、代表銘柄ユニバースを通じて各国株価指数先物へ集約するPython実装です。

v0.4では、約2,500銘柄/時点のデータを長期間処理することを想定し、以下を追加しました。

- サマリーExcelから履歴データを分離
- 履歴の種類ごとに1つのExcelファイルを出力
- 1,048,576行のExcel上限を考慮したシート自動分割
- サマリーExcel・各履歴Excel・各PDFの出力可否をConfigで個別指定
- 4候補モデルの比較結果、1-SE閾値、採用理由を明示
- 全候補モデルのローリングOOS RankICをPDFで比較
- モデル選択専用PDFを追加

ファクター固有の定義は `factor_master.xlsx` で管理し、モデル全体の共通設定と出力制御は `config/model_config.py` で管理します。

---

## 2. 設定の役割分担

```text
factor_master.xlsx
    ├─ FA0101等のファクターコードと名称
    ├─ Value / Momentum / Quality等の所属グループ
    ├─ Direction、有効・無効、手動ウェイト
    ├─ ファクター固有の前処理例外
    └─ グループごとの統合方法

model_config.py
    ├─ 入出力ファイル
    ├─ 分析頻度
    ├─ 共通の前処理デフォルト
    ├─ 学習・検証期間
    ├─ 単一ファクターモデル選択基準
    ├─ 代表銘柄数と選定条件
    └─ サマリーExcel・履歴Excel・PDFの出力可否
```

実行時の設定優先順位は次のとおりです。

```text
Factor_Masterの個別設定
    > Group_Settingsのグループ設定
        > model_config.pyの共通デフォルト
```

---

## 3. フォルダ構成

```text
stock_index_scoring_model_v4/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ config/
│  └─ model_config.py
├─ data/
│  ├─ input/
│  │  ├─ factor_master.xlsx
│  │  ├─ factors_and_returns.xlsx
│  │  ├─ index_constituents.xlsx
│  │  ├─ index_sector_weights.xlsx
│  │  └─ futures_returns.xlsx
│  └─ templates/
├─ docs/
│  ├─ data_dictionary.md
│  ├─ factor_master_guide.md
│  ├─ methodology.md
│  ├─ model_selection.md
│  ├─ output_management.md
│  └─ representative_universe.md
├─ outputs/
│  ├─ analysis_summary.xlsx
│  ├─ factor_scatter_diagnostics.pdf
│  ├─ factor_bin_diagnostics.pdf
│  ├─ factor_model_selection_report.pdf
│  ├─ factor_model_performance.pdf
│  ├─ index_factor_exposure.pdf
│  ├─ index_factor_trends.pdf
│  ├─ model_accuracy_report.pdf
│  ├─ universe_selection_report.pdf
│  ├─ futures_risk_report.pdf
│  └─ history/
│     ├─ index_score_history.xlsx
│     ├─ index_factor_history.xlsx
│     ├─ factor_performance.xlsx
│     └─ ...
├─ scripts/
│  ├─ run_pipeline.py
│  └─ generate_demo_data.py
├─ src/stock_index_model/
└─ tests/
```

---

## 4. 入力ファイル

### 4.1 `factor_master.xlsx`

最初のシートはREADMEです。

#### `Factor_Master`

| 列 | 内容 |
|---|---|
| `Factor_Code` | 元データの列名。例：FA0101 |
| `Factor_Name_JP` / `Factor_Name_EN` | 表示名称 |
| `Factor_Group` | Value、Momentum、Qualityなど |
| `Enabled` | 1なら使用、0なら停止 |
| `Direction` | 1：高いほど望ましい、-1：低いほど望ましい |
| `Base_Weight` | manual統合時の基準ウェイト |
| `Transform` | default、none、log、log1p、inverse、signed_log |
| `Winsorize` | default、none、1_99、2.5_97.5、mad_3 |
| `Neutralize` | default、1、0 |
| `Rank_Normalize` | default、1、0 |
| `Min_Coverage` | 利用する最低観測率 |

#### `Group_Settings`

グループごとに統合方法を指定します。

| `Aggregation_Method` | 内容 |
|---|---|
| `equal_weight` | 有効ファクターを等ウェイトで統合 |
| `manual` | Base_Weightを正規化して統合 |
| `ic_adjusted` | 過去RankICとファクター相関を考慮 |
| `pca` | 共通変動を表す主成分で統合 |

### 4.2 `factors_and_returns.xlsx`

標準シート名は `data` です。

| 列 | 内容 |
|---|---|
| `date` | ファクター観測時点 |
| `ISIN` | 個別銘柄キー |
| `stock_return` | 当該時点から翌期までの個別銘柄リターン |
| `sector` | セクター |
| `country` | 国または地域 |
| `market_cap` | 時価総額 |
| `FAxxxx` | Factor_Masterで定義したファクター列 |

### 4.3 `index_constituents.xlsx`

- 1指数につき1シート
- シート名が指数名
- 各シートに `ISIN`、`sector`、必要に応じて `country` を格納

### 4.4 `index_sector_weights.xlsx`

縦にセクター、横に指数名を並べます。値は0-1または0-100で入力できます。

### 4.5 `futures_returns.xlsx`

- `monthly_returns`
- `weekly_returns`

のいずれを使用するかは `CONFIG["data"]["frequency"]` で指定します。

---

## 5. 全体処理フロー

```text
1. 入力Excelとfactor_master.xlsxの読込
2. マスタ・入力列・グループ設定の整合性検証
3. 代表銘柄ユニバースのローリング選定
4. 個別ファクターの前処理
5. ファクタービン分析
6. 4候補モデルのWalk-forward OOS推定
7. OOS平均RankICと1-SE ruleによる単一ファクターモデル選択
8. Excel指定方法によるグループスコア統合
9. OOF Ridgeによるグループ間統合
10. 個別銘柄alpha・順位・信頼度スコア作成
11. 代表銘柄ウェイトで指数へ集約
12. 指数ファクター傾向・寄与・Breadth算出
13. 指数スコアと翌期先物リターンのOOS評価
14. 当月・ローリング先物リスク算出
15. サマリーExcel・履歴Excel・PDFを設定に従って出力
```

---

## 6. 単一ファクターモデルの選択

候補は次の4つです。

1. `linear`
2. `piecewise`
3. `quadratic`
4. `combined_ridge`

### 6.1 判定フロー

```text
候補4モデルをWalk-forwardでOOS評価
    ↓
各モデルの月次OOS RankICを計算
    ↓
平均RankICが最大のモデルをbest_raw_modelとする
    ↓
1-SE閾値 = best平均RankIC - bestのRankIC標準誤差
    ↓
閾値以上のモデルを「最良モデルとほぼ同等」と判定
    ↓
ほぼ同等候補の中から最も単純なモデルを採用
```

既定の複雑度は次のとおりです。

| モデル | 複雑度 |
|---|---:|
| Linear | 1 |
| Piecewise | 2 |
| Quadratic | 2 |
| Combined Ridge | 3 |

### 6.2 Linearが多く選ばれる理由

Linearが無条件に優先されるわけではありません。

- Linear自体の平均OOS RankICが最も高い場合
- 非線形モデルの平均OOS RankICが少し高くても、その差が1標準誤差以内の場合

にLinearが選ばれます。

後者は、非線形モデルの改善が推定誤差の範囲内であり、将来安定性と説明可能性を優先して単純モデルを採用したことを意味します。Linearが1-SE閾値を下回り、非線形モデルが明確に優れる場合はPiecewise、Quadratic、Combined Ridgeが選択されます。

### 6.3 確認する出力

- `Factor_Model_Selection`：ファクターごとの最終判定
- `Factor_Model_Candidate_Summary`：4候補すべての比較
- `Factor_Model_Methodology`：判定フローの説明
- `factor_model_selection_report.pdf`：平均RankIC、標準誤差、1-SE閾値の可視化
- `factor_model_performance.pdf`：4候補のローリングOOS RankIC推移

詳細は [`docs/model_selection.md`](docs/model_selection.md) を参照してください。

---

## 7. 大容量履歴の出力設計

約2,500銘柄/時点の個別銘柄履歴を1つのブックにまとめると、ファイルサイズ・保存時間・Excel操作性が悪化します。

v0.4では次の方針です。

- `analysis_summary.xlsx`：最新値、要約、設定、モデル判定だけ
- `outputs/history/*.xlsx`：履歴の種類ごとに1ファイル
- 各履歴ファイル：`README` + `Data_001`、`Data_002` ...
- 行数が `max_rows_per_sheet` を超えた場合、同じExcel内で自動分割
- 個別銘柄の巨大履歴は初期状態で出力しない

### 7.1 履歴出力の切替

```python
"history_excel": {
    "enabled": True,
    "output_subdir": "history",
    "max_rows_per_sheet": 800000,
    "tables": {
        "Index_Score_History": True,
        "Index_Factor_History": True,
        "Factor_Performance": True,
        "Group_Score_History": False,
        "Stock_Score_History": False,
    },
}
```

`False` の履歴ファイルは作成されません。

### 7.2 PDF出力の切替

```python
"pdf": {
    "enabled": True,
    "reports": {
        "factor_scatter": True,
        "factor_bin": True,
        "factor_model_selection": True,
        "factor_model_performance": True,
        "index_factor_trends": False,
    },
}
```

### 7.3 サマリーExcelのシート切替

```python
"summary_excel": {
    "enabled": True,
    "filename": "analysis_summary.xlsx",
    "sheets": {
        "Stock_Scores_Latest": True,
        "Group_Scores_Latest": False,
    },
}
```

未指定のサマリーシートは既定で出力します。

詳細は [`docs/output_management.md`](docs/output_management.md) を参照してください。

---

## 8. 主な出力

### 8.1 サマリーExcel

`outputs/analysis_summary.xlsx`

先頭のREADMEに各シートの内容を記載しています。主なシートは次のとおりです。

- `Output_Manifest`
- `Executive_Summary`
- `Factor_Model_Selection`
- `Factor_Model_Candidate_Summary`
- `Factor_Model_Methodology`
- `Index_Scores_Latest`
- `Index_Factor_Exposure`
- `Model_Accuracy_Summary`
- `Futures_Risk_Latest`

### 8.2 履歴Excel

`outputs/history/` 以下に1種類1ファイルで出力します。

例：

- `index_score_history.xlsx`
- `index_factor_history.xlsx`
- `factor_performance.xlsx`
- `factor_coefficients.xlsx`
- `model_accuracy_history.xlsx`

### 8.3 PDF

| PDF | 内容 |
|---|---|
| `factor_scatter_diagnostics.pdf` | 散布図と候補回帰線 |
| `factor_bin_diagnostics.pdf` | 分位別翌期リターン |
| `factor_model_selection_report.pdf` | 候補比較、1-SE閾値、採用理由 |
| `factor_model_performance.pdf` | 全候補モデルのローリングOOS RankIC |
| `index_factor_exposure.pdf` | 最新指数スコア・ファクター傾向 |
| `index_factor_trends.pdf` | 指数ファクター傾向時系列 |
| `model_accuracy_report.pdf` | 指数モデルの正解率推移 |
| `universe_selection_report.pdf` | 代表ユニバース選定品質 |
| `futures_risk_report.pdf` | 指数先物リスク |

---

## 9. 実行方法

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
```

macOS / Linux：

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --config config/model_config.py
```

テスト：

```bash
pytest -q
```

---

## 10. 実運用時の注意

- 財務ファクターはPoint-in-Timeデータを使用してください。
- `stock_return`はファクター観測時点から翌期までのリターンに揃えてください。
- Factor_Masterの変更履歴をGit、SharePoint等で管理してください。
- `Output_Manifest`で有効化されている出力と行数を確認してください。
- 大容量の `Group_Score_History` と `Stock_Score_History` は、必要な分析時だけ有効にしてください。
- 1-SE ruleを無効化すると、平均RankIC最大のモデルを直接採用しますが、複雑モデルの採用が増えやすくなります。
- 複雑なモデルが等ウェイト・Linear等の単純ベースラインをOOSで明確に上回るかを必ず確認してください。
