# 出力容量管理

## パターン別Excel

1パターンにつき1ファイルです。横持ちのStockScoreは8列程度に限定し、SubScoreとFactorScoreは縦持ちで保存します。

デフォルトは `date_scope="latest"` です。全履歴が必要な場合のみ `all` に変更します。

## 履歴

履歴は種類ごとに別Excelへ出力します。個別銘柄の全履歴は大容量になるため、`Group_Score_History` と `Stock_Score_History` は初期設定でFalseです。

## PDF

PDFは種類ごとにConfigでON/OFFを設定できます。個別銘柄モデルの目視評価では、少なくとも次の2つを有効にすることを推奨します。

- `scenario_quintile_cumulative`
- `scenario_comparison`
