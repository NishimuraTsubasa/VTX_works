# 出力仕様

## パターン別Excel

1パターンにつき1ファイルです。

- `StockScore_001`：TotalScore、Prediction、翌月リターン、時価総額、通貨等
- `SubScore_001`：FA単位の第1層SubScore（縦持ち）
- `FactorScore_001`：Factor_Group単位の第2層FactorScore（縦持ち）
- `FactorWeights`：第2層ウェイト
- `ModelSelection`：第1層候補モデル選択

既定は最新時点のみ出力します。

## PDF

- `quintile_cumulative_returns.pdf`
- `stock_scoring_scenario_comparison.pdf`
- `layer3_scope_comparison.pdf`
- `layer3_country_diagnostics.pdf`
- `coefficient_stability.pdf`
- `sector_factor_interactions.pdf`
- Binscatter 3種類

## 履歴Excel

大容量出力は個別ファイルへ分離します。ConfigでON/OFFを指定できます。

## v0.12.6 共通OOS・S07推定方式比較

### analysis_summary.xlsx

- `Scenario_Summary`: 全期間指標と共通OOS指標
- `Common_Quintiles`: 共通Date×ISIN上で再計算した5分位リターン
- `Common_RankIC`: 共通OOSの月次RankIC
- `Common_RankIC_Delta`: S03対比のMean/Median RankIC差と勝率

### s07_ols_ridge_comparison.xlsx

- `Summary`: OLS/Ridgeの共通OOS指標
- `Common_RankIC`: 月次RankIC
- `Common_Quintiles`: 共通OOS分位リターン
- `Model_History`: estimator、alpha、学習期間、検証期間、観測数
- `Coefficients`: 第3層係数履歴

### s07_ols_ridge_comparison.pdf

1. 共通OOS Q5-Q1累積リターン
2. 共通OOS RankIC推移
3. Mean RankIC・ICIR等の比較
