# 出力管理

## 基本方針

- 最新値・要約・設定は `analysis_summary.xlsx`
- 履歴は `outputs/history` 以下へ種類ごとに1ファイル
- PDFはレポート種類ごとに個別制御

## Excel行上限への対応

`report.history_excel.max_rows_per_sheet` を超えた場合、同じ履歴ファイル内で次のように分割します。

```text
README
Data_001
Data_002
...
```

## 推奨初期設定

個別銘柄数が約2,500銘柄の場合、次は初期値Falseを推奨します。

- `Group_Score_History`
- `Stock_Score_History`
- `Universe_Selection_History`（代表銘柄数・期間による）

指数単位、ファクター単位の履歴は比較的小さいためTrueで問題ありません。

## 出力の監査

サマリーExcelの `Output_Manifest` に、以下を記録します。

- 出力ID
- 種類
- 有効/無効
- 生成結果
- 行数・列数
- 相対ファイルパス
- 内容
