# 入力生成README

## factors_and_returns.xlsx

`data`シートの1行は「日付 × ISIN」です。

| Column | 内容 | 型 | 注意 |
|---|---|---|---|
| date | ファクター観測時点 | Date | 月末等で統一 |
| ISIN | 銘柄キー | Text | dateとの組合せで一意 |
| stock_return | 当該行時点のリターン | Decimal | Configで翌月化 |
| market_cap | 時価総額 | Decimal | 正値 |
| currency | 通貨 | Text | 出力用 |
| country | 国 | Text | Country_Region_Mapと一致 |
| sector | セクター | Text | Sector_Group_Mapと一致 |
| FAxxxx | ファクター値 | Decimal | 欠損は空欄、0埋めしない |

## 時点管理

- Raw FA：`x_t -> r_(t+1)`
- Source_Lag=1の派生FA：`x_(t-1)`以前から作成し、`r_(t+1)`を予測
- 第1層の過去SubScoreは必ずOOF

## factor_master.xlsx

通常編集するシート：

- Factor_Master
- Group_Settings
- Feature_Engineering_Control
- Derived_Feature_Rules
- Country_Region_Map
- Sector_Group_Map
- Sector_Factor_Interaction
- Layer3_Settings

詳細はExcel内の `Column_Dictionary` と `Option_Dictionary` を参照してください。
