# factor_master.xlsx リファレンス

## v0.12追加シート

### Country_Region_Map

| Column | 内容 |
|---|---|
| Country | factors_and_returns.xlsxのcountryと完全一致 |
| Region | 地域プール・部分プーリングの単位 |
| Enabled | 1=使用、0=無効 |

### Sector_Group_Map

| Column | 内容 |
|---|---|
| Sector | 入力sectorと完全一致 |
| Sector_Group | 第3層で使うグループ |
| Enabled | 1=使用 |

### Sector_Factor_Interaction

`Interaction_Mode=selected_interactions`のとき、Enabled=1の組合せだけ交差項を作ります。

### Layer3_Settings

| Setting | 選択肢 |
|---|---|
| Estimation_Scope | country_independent / regional_pooling / hierarchical_partial_pooling |
| Training_Mode | rolling_pooled / cross_sectional_coefficient_average |
| Interaction_Mode | selected_interactions / all_interactions |
| Include_Nonlinear_Basis | 0 / 1 |
| Include_Sector_Dummy | 0 / 1 |
| Include_Sector_Factor_Interaction | 0 / 1 |

Excel設定が空欄の場合、model_config.pyを使用します。
