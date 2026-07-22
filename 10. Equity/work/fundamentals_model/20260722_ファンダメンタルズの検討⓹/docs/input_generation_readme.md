# 入力生成README

## factors_and_returns.xlsx

`data`シートの1行目に次のカラムを置きます。

| Column | 内容 |
|---|---|
| date | スコア形成時点 |
| ISIN | 銘柄キー |
| stock_return | 当月リターン。Configで翌月へシフト |
| market_cap | 時価総額 |
| currency | 通貨 |
| country | 国 |
| sector | セクター |
| FAxxxx | Raw Factor値 |

- date+ISINは一意
- 欠損を0で埋めない
- Factor_Codeはfactor_master.xlsxと完全一致
- market_capは正値を推奨

## factor_master.xlsx

README以外は1行目ヘッダー、2行目以降データです。

第1層回帰関連の設定シートはv0.13.1では削除されています。
