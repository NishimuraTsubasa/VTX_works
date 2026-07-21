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
