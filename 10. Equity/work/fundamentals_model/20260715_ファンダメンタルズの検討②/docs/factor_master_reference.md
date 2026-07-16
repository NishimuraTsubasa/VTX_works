# factor_master.xlsx リファレンス

本ファイルは、個別銘柄スコアリングモデルのファクター定義、グループ統合、派生特徴量、例外設定を管理します。

## 主要シート

| Sheet | 用途 |
|---|---|
| `README` | 更新手順・注意事項 |
| `Column_Dictionary` | 全カラムの詳細定義 |
| `Option_Dictionary` | 選択式項目の選択肢と意味 |
| `Factor_Master` | FAコード、グループ、使用可否、方向、固定ウェイト |
| `Group_Settings` | グループの使用可否と統合方法 |
| `Feature_Engineering_Control` | 派生特徴量の生成対象 |
| `Derived_Feature_Rules` | 差分・移動平均乖離等の生成ルール |
| `Factor_Overrides` | FA固有の前処理例外 |
| `Group_Overrides` | グループ固有の推定設定例外 |

詳細は [`factor_master_excel_creation_instructions.txt`](factor_master_excel_creation_instructions.txt) を参照してください。
