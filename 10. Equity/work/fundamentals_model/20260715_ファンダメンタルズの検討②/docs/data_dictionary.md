# データ辞書

## factors_and_returns.xlsx / data

| 列 | 型 | 内容 |
|---|---|---|
| date | date | スコア計算時点 |
| ISIN | string | 銘柄識別子 |
| stock_return | float | 個別銘柄リターン |
| market_cap | float | 時価総額 |
| currency | string | 通貨コード |
| country | string | 国 |
| sector | string | セクター |
| FAxxxx | float | ファクター値 |

`stock_return_alignment="contemporaneous_to_forward"` の場合、内部で銘柄ごとに1期先へシフトして `forward_return` を作成します。

## パターン別StockScore

| 列 | 内容 |
|---|---|
| Date | スコア計算日 |
| ISIN | 銘柄識別子 |
| Currency | 通貨 |
| MarketCap | 時価総額 |
| TotalScore | 0-1の最終個別銘柄スコア |
| Prediction | 順位化前予測シグナル |
| NextMonthReturn | 翌期リターン |
| Quintile | 1から5のスコア分位 |

## SubScore

| 列 | 内容 |
|---|---|
| Date | スコア計算日 |
| ISIN | 銘柄識別子 |
| SubScore | Value、Momentum等 |
| SubScoreValue | サブスコア値 |

## FactorScore

| 列 | 内容 |
|---|---|
| Date | スコア計算日 |
| ISIN | 銘柄識別子 |
| FactorCode | FA0101等 |
| FactorScore | ファクター別スコア |
