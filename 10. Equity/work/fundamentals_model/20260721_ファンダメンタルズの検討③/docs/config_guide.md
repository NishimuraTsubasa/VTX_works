# Config設定ガイド

## 通常変更する場所

### layer1

- `training_window_periods`：単一FAモデルの学習窓
- `minimum_train_periods`：学習開始の最低期間
- `candidate_models`：通常はlinear / piecewise / quadratic
- `one_se_rule`：誤差範囲内なら単純モデルを選択

### layer2

- `ic_lookback_periods`
- `maximum_factor_weight`
- `weight_smoothing`

集約方式自体は `factor_master.xlsx` の `Group_Settings` で管理します。

### layer3

- `primary_scope`
- `comparison_scopes`
- `training_mode`
- `lookback_periods`
- `nonlinear_basis`
- `interaction_mode`
- `ridge_alphas`
- `final_score_rank_scope`

### 推奨初期値

```python
"primary_scope": "country_independent"
"comparison_scopes": ["country_independent", "regional_pooling"]
"training_mode": "rolling_pooled"
"lookback_periods": 36
"interaction_mode": "selected_interactions"
"final_score_rank_scope": "country"
```

`hierarchical_partial_pooling` は計算量が増えるため、比較時に `comparison_scopes` へ追加します。
