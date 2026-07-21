# v0.12.5 FactorScore予測力診断・日本語PDF対応パッチ

## 前提

本パッチは、`v0.12.4_model_diagnostics_patch` 適用後のプロジェクトへ上書きしてください。
本ZIPには、今回修正・新規追加したコードだけを収録しています。

## 追加出力

パイプライン実行後、次の2ファイルを生成します。

```text
outputs/
├─ factor_score_performance_diagnostics.xlsx
└─ factor_score_performance_diagnostics.pdf
```

## 分析目的

次の各段階で、高スコア銘柄が実際に高い翌月リターンを得たかを確認します。

```text
回帰前Input Score
→ OOF SubScore
→ 集約FactorScore
→ S06
→ S07
```

評価サンプルは、初期設定では以下の3シナリオが共通して予測値を持つ `Date × ISIN` の積集合です。

```text
S06_Selected_Factor_Models
S07_OLS_Linear
S07_Ridge_Linear
```

## Excelシート

| シート | 内容 |
|---|---|
| `FactorGroup_Summary` | FactorScore別のMean Rank IC、ICIR、Q5-Q1、単調性 |
| `FactorGroup_MonthlyIC` | グローバル・国別内・国×セクター内の月次Rank IC |
| `FactorGroup_Quintiles` | FactorScore別Q1-Q5翌月リターン |
| `FactorGroup_LongShort` | FactorScore別Q5-Q1時系列と累積値 |
| `FactorGroup_Calibration` | Time-Averaged Bin別のスコアと実現リターン |
| `SubScore_Summary` | 単一FAのOOF SubScore予測力 |
| `SubScore_MonthlyIC` | 単一FAの月次Rank IC |
| `SubScore_Quintiles` | 単一FAのQ1-Q5翌月リターン |
| `Raw_vs_SubScore` | 回帰前Input ScoreとOOF SubScoreの比較 |
| `Country_FactorGroup` | 国別FactorScore予測力 |
| `CountrySector_FactorGroup` | 国×セクター別FactorScore予測力 |
| `FactorGroup_Correlation` | FactorScore間の平均Spearman相関 |
| `LeaveOneGroupOut` | S06から各Factor Groupを除外した増分分析 |
| `Coverage_Dispersion` | FactorScoreのカバレッジと分散推移 |
| `Common_OOS_Keys` | 共通OOSのDate×ISIN |

## PDFページ

- FactorScore別Mean Rank ICとQ5-Q1
- FactorScore別ローリングRank IC
- FactorScore別Q1-Q5翌月リターン
- FactorScore別Q5-Q1累積リターン
- FactorScoreのTime-Averaged Binscatter
- 国×FactorScoreのMean Rank ICヒートマップ
- FactorScore間相関ヒートマップ
- Leave-One-Group-Out増分分析
- 回帰前Input ScoreとOOF SubScoreのRank IC変化
- OOF SubScoreの上位・下位Factor

## 日本語PDF対応

`reporting.py` のフォント設定をOS横断方式へ変更しました。
実行PCにインストールされている次の日本語フォントから自動選択します。

```text
Noto Sans CJK JP
Noto Sans JP
Yu Gothic / YuGothic
Meiryo
MS Gothic
IPAexGothic / IPAGothic
Hiragino Sans
TakaoGothic
```

また、PDFへTrueTypeフォントを埋め込むため、`pdf.fonttype=42` を設定しています。
フォントファイル自体は同梱していません。

Windowsで文字化けする場合は、通常は `Yu Gothic` または `Meiryo` が利用されます。

## Config追加項目

```python
"factor_score_performance_diagnostics": {
    "common_oos_scenarios": [
        "S06_Selected_Factor_Models",
        "S07_OLS_Linear",
        "S07_Ridge_Linear",
    ],
    "minimum_stocks_per_date": 30,
    "minimum_stocks_per_group": 15,
    "minimum_stocks_per_country_sector": 8,
    "minimum_country_sector_periods": 6,
    "quantiles": 5,
    "calibration_bins": 10,
    "rolling_rank_ic_periods": 12,
    "subscore_top_n_pdf": 12,
},
```

出力スイッチも追加しています。

```python
"factor_score_performance_diagnostics_xlsx": True
"pdf": {
    "factor_score_performance_diagnostics": True,
}
```

## 差し替え手順

ZIPを展開し、中身を既存プロジェクトルートへ同じ階層でコピーして上書きします。

```powershell
py -m pip install -e .
py -m pytest -q
py scripts\run_pipeline.py --config config\model_config.py
```

## 追加テスト

```text
tests/test_factor_score_performance_reporting.py
tests/test_japanese_pdf_font.py
```

開発環境では、既存テストを含め21件すべて成功しています。
