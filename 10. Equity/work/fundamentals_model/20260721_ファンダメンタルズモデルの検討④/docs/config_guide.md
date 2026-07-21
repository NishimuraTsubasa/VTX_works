# Config設定ガイド v0.12.6

## 1. 2018年開始データの推奨初期値

```python
"layer1": {
    "training_window_periods": 36,
    "minimum_train_periods": 18,
    "validation_periods": 6,
},
"layer3": {
    "lookback_periods": 36,
    "minimum_train_periods": 12,
    "ridge_validation_periods": 6,
},
```

第3層の `minimum_train_periods` は、元データの2018年から数えるのではなく、第1層のOOF FactorScoreが利用可能になった後の月数です。

## 2. 共通OOS

```python
"evaluation": {
    "common_oos": {
        "enabled": True,
        "universe_mode": "stock_date_intersection",
        "rerank_on_common_universe": True,
        "benchmark_scenario": "S03_Neutralized_Direct_EW",
        "minimum_stocks_per_date": 30,
        "minimum_periods_warning": 24,
    }
}
```

### stock_date_intersection

全シナリオで有効な `Date × ISIN` の積集合を使用します。期間だけでなく銘柄集合も揃えるため、最も公平な比較です。

### rerank_on_common_universe

共通集合へ限定した後、各シナリオの順位と5分位を再計算します。元の広いユニバースで計算した順位を流用しません。

## 3. S07推定方式

```python
"s07_variants": {
    "S07_OLS_Linear": {
        "enabled": True,
        "estimator": "ols",
        "nonlinear_basis": ["linear"],
    },
    "S07_Ridge_Linear": {
        "enabled": True,
        "estimator": "ridge",
        "nonlinear_basis": ["linear"],
    },
    "S07_Ridge_Flexible": {
        "enabled": False,
        "estimator": "ridge",
        "nonlinear_basis": ["linear", "piecewise", "quadratic"],
    },
}
```

OLSとRidgeの正則化効果だけを比較するため、最初の2つは同じ線形基底を使用します。3基底Ridgeは補助分析です。

## 4. Ridge alphaの選択

```python
"ridge_alphas": [0.1, 1.0, 10.0],
"ridge_validation_periods": 6,
```

各予測時点で、過去学習窓の末尾6か月をalpha選択用に使用し、選択後に全過去データで再推定します。評価対象となる将来月はalpha選択に使用しません。

## 5. 第3層推定範囲

```python
"primary_scope": "country_independent",
"comparison_scopes": [
    "country_independent",
    "regional_pooling",
],
```

- `country_independent`：国ごとに独立推定
- `regional_pooling`：同一地域の国をまとめる
- `hierarchical_partial_pooling`：地域共通係数と国固有補正を同時推定

S07のOLS/Ridge本線比較は `primary_scope` で実施し、推定範囲比較は別の診断出力で行います。

## 6. 学習期間の感応度分析

2018年開始データでは次の候補を比較してください。

```text
Layer1 minimum_train_periods : 12 / 18 / 24
Layer3 minimum_train_periods : 9 / 12 / 18
Lookback periods             : 24 / 36 / 48 / expanding相当
Ridge validation periods     : 3 / 6 / 9
```

短い学習期間は共通OOSを長くできますが、係数・モデル選択が不安定になる可能性があります。主比較は同一共通OOS期間で行い、係数安定性と合わせて判断します。

## 7. Excel側設定

`factor_master.xlsx` の `Layer3_Settings` でも次を設定できます。

| Setting | 初期値 |
|---|---:|
| Lookback_Periods | 36 |
| Minimum_Train_Periods | 12 |
| Ridge_Validation_Periods | 6 |
| S07_OLS_Linear_Enabled | 1 |
| S07_Ridge_Linear_Enabled | 1 |
| S07_Ridge_Flexible_Enabled | 0 |
