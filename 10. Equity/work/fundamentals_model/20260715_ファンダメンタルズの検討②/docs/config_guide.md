# Config設定ガイド

設定ファイルは`config/model_config.py`です。

## 1. データ

```python
"data": {
    "factors_file": "data/input/factors_and_returns.xlsx",
    "factors_sheet": "data",
    "factor_master_file": "data/input/factor_master.xlsx",
    "frequency": "monthly",
}
```

## 2. 目的変数

```python
"target": {
    "stock_return_alignment": "contemporaneous_to_forward",
    "stock_horizon_periods": 1,
}
```

## 3. シナリオ

| ID | 内容 |
|---|---|
| S00 | 現行: 0-1順位を直接等ウェイト。欠損は中立値0.5 |
| S01 | 欠損FAを除いて再正規化 |
| S02 | Winsorize後に直接等ウェイト |
| S03 | 国・セクター・サイズ中立化後に直接等ウェイト |
| S04 | グループ内EW + グループ間EW |
| S05 | グループ内相関調整IC |
| S06 | 4候補モデルを時系列OOSで選択 |
| S07 | グループ予測をOOF Ridgeで統合 |

Trueにしたものだけ計算します。

## 4. Binscatter対象FA

```python
"factor_codes": ["FA0101", "FA0102"]
```

空リストの場合、Enabledな元FA・派生FAを全て対象にします。

## 5. スコープ

```python
"scopes": {
    "all_universe": True,
    "by_country": True,
    "by_country_sector": True,
}
```

サンプルでは国・国 x セクター数を限定しています。実データで全件にする場合:

```python
"countries": [],
"sectors": [],
"max_country_sector_scopes": 0,
```

## 6. ビン数

```python
"n_bins": {
    "all_universe": 20,
    "by_country": 20,
    "by_country_sector": 10,
}
```

国 x セクターは銘柄数が少なくなるため、10ビンを推奨します。

## 7. 回帰

```python
"regressions": {
    "linear": True,
    "quadratic": True,
    "broken_stick": True,
    "broken_stick_knot": "auto",
}
```

`broken_stick_knot`:

- `auto`: 候補knotの中からビン点へのR2最大
- `zero`: 0固定
- `median`: xの中央値

## 8. エラーバー

```python
"error_bar": "standard_error"
```

選択肢:

- `standard_error`
- `ci95`
- `none`

## 9. 出力制御

```python
"outputs": {
    "analysis_summary_xlsx": True,
    "scenario_excel": {
        "enabled": True,
        "date_scope": "latest",
        "include_sub_scores": True,
        "include_factor_scores": True,
    },
    "pdf": {
        "binscatter_all_universe": True,
        "binscatter_by_country": True,
        "binscatter_by_country_sector": True,
        "quintile_cumulative_returns": True,
        "scenario_comparison": True,
    },
}
```

2,500銘柄の全履歴を出すと容量が大きくなるため、パターン別Excelは`latest`を標準としています。
