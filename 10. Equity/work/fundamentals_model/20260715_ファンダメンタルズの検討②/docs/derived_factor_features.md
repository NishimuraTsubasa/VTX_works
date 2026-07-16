# 差分・移動平均乖離ファクターの設定と時点整合

## 1. 目的

原系列ファクターだけでなく、ファクターの変化や通常水準からの乖離を説明変数として追加します。

例：

- 1期差分
- 過去移動平均からの乖離
- 過去移動平均に対する比率
- 全過去平均からの乖離

## 2. 時点整合

スコア計算時点を `t`、評価対象の翌期リターンを `r[t+1]` とします。

原系列は `x[t]` を使用するため、最新ファクター観測からターゲットまでの間隔は1期です。

派生特徴量の標準設定では `Source_Lag_Periods = 1` とし、スコア時点 `t` の行へ次を格納します。

```text
1期差分 = x[t-1] - x[t-2]
12期移動平均乖離 = x[t-1] - mean(x[t-2], ..., x[t-13])
```

この特徴量を `r[t+1]` と対応させるため、最新の使用情報 `x[t-1]` からターゲット `r[t+1]` までの実効間隔は2期です。

```text
Effective_Target_Gap_Periods
= stock_horizon_periods + Source_Lag_Periods
= 1 + 1
= 2
```

## 3. Excel設定

### Feature_Engineering_Control

| 列 | 内容 |
|---|---|
| Scope_Type | `group` または `factor` |
| Scope_Value | `Value`、`Momentum`、`FA0101`等 |
| Enabled | 派生特徴量生成を有効化 |
| Generation_Mode | `all` または `selected` |
| Include_Raw | 原系列も説明変数として残すか |

個別ファクター指定はグループ指定より優先します。

### Derived_Feature_Rules

| Feature_Type | 式 |
|---|---|
| difference | ラグ済み原系列の差分 |
| rolling_mean_deviation | ラグ済み原系列 - 過去移動平均 |
| rolling_mean_ratio | ラグ済み原系列 / 過去移動平均 - 1 |
| expanding_mean_deviation | ラグ済み原系列 - 全過去平均 |

`Generation_Mode=all` の場合は、Enabledなルールをすべて使用します。

`Generation_Mode=selected` の場合は、EnabledかつSelected=1のルールだけを使用します。

## 4. 派生ファクターコード

自動生成例：

```text
FA0101                         原系列
FA0101__DIFF_P1_L1            1期差分、Source Lag=1
FA0101__MADEV_W12_L1          12期移動平均乖離、Source Lag=1
FA0101__EXPDEV_L1             過去平均乖離、Source Lag=1
```

各派生ファクターは独立した説明変数として、外れ値処理、中立化、順位変換、ビン分析、4候補モデル比較の対象になります。

## 5. 出力確認

`analysis_summary.xlsx` の以下を確認します。

- `Feature_Lineage`: 生成式、元ファクター、情報ラグ、実効ターゲット間隔
- `Feature_Engineering_Control`: 適用した生成モード
- `Derived_Feature_Rules`: 使用可能な生成ルール
- `Factor_Model_Selection`: 原系列・派生系列ごとの採用モデル
- `Factor_Bin_Factor_Summary`: 派生系列を含む5分位有効性

パターン別Excelの `Factor_Map` と `FactorScore` にも派生ファクターが表示されます。
