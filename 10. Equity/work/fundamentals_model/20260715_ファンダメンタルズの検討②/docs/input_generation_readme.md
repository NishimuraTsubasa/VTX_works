# インプット生成README

## 1. `factors_and_returns.xlsx`

### データ粒度

```text
1行 = 1時点 x 1銘柄
主キー = date + ISIN
```

同じ`date + ISIN`が複数行ある場合、処理を停止します。

## 2. 必須カラム

| カラム | 型 | 内容 | 例 | 注意点 |
|---|---|---|---|---|
| `date` | 日付 | ファクター観測時点 | `2025-01-31` | 月次なら原則月末日 |
| `ISIN` | 文字列 | 銘柄キー | `US0000000001` | 前後空白を入れない |
| `stock_return` | 数値 | 入力行の日付に対応するリターン | `0.025` | 2.5%は`0.025` |
| `market_cap` | 数値 | 予測時点の時価総額 | `1.5e11` | 正の値を使用 |
| `currency` | 文字列 | 通貨コード | `USD` | 出力・集計用 |
| `country` | 文字列 | 国・市場区分 | `US` | 国別binscatterで使用 |
| `sector` | 文字列 | セクター | `Technology` | 国 x セクター分析で使用 |
| `FAxxxx` | 数値 | ファクター生値 | `0.42` | 欠損は空欄/NaN |

## 3. `stock_return`の時点定義

### 当月リターンを入力する場合

```python
"stock_return_alignment": "contemporaneous_to_forward"
```

入力例:

```text
2025-01-31行のstock_return = 2025年1月リターン
```

内部では、2025-01-31のファクターに2025年2月リターンを対応させます。

### すでに翌期リターンを入力する場合

```python
"stock_return_alignment": "already_forward"
```

入力例:

```text
2025-01-31行のstock_return = 2025年2月リターン
```

この設定を誤ると目的変数がさらに1期ずれるため、初回実行前に必ず確認してください。

## 4. Point-in-Timeルール

- 財務データは決算期ではなく、予測時点で利用可能だった値を使用
- 修正後データを過去時点へ遡及適用しない
- 上場廃止銘柄を過去ユニバースから除外しない
- 株式分割、通貨、単位を整合させる

## 5. FAカラム

FAコードだけで管理します。

```text
FA0101
FA0102
FA1001
```

名称列は不要です。`factor_master.xlsx`の`Factor_Code`と完全一致させます。

## 6. 欠損値

- ファクター欠損を0で埋めて入力しない
- 空欄またはNaNを使用
- 欠損の扱いはシナリオ側で実施
- `market_cap`、`country`、`sector`が欠損すると中立化・スコープ分析に影響

## 7. `factor_master.xlsx`

### `Factor_Master`

| カラム | 内容 |
|---|---|
| `Factor_Code` | FA列名 |
| `Factor_Group` | Value、Momentum、Quality等 |
| `Enabled` | 1=使用、0=不使用 |
| `Direction` | 1=高いほど良い、-1=低いほど良い |
| `Base_Weight` | manual統合時の相対ウェイト |

### `Group_Settings`

| カラム | 内容 |
|---|---|
| `Factor_Group` | グループ名 |
| `Enabled` | 1=使用、0=不使用 |
| `Aggregation_Method` | `equal_weight` / `manual` / `ic_adjusted` / `pca` |

### `Feature_Engineering_Control`

| カラム | 内容 |
|---|---|
| `Scope_Type` | `group` または `factor` |
| `Scope_Value` | Value またはFA0101等 |
| `Enabled` | 派生生成のON/OFF |
| `Generation_Mode` | `all` / `selected` |
| `Include_Raw` | 元FAも説明変数に残すか |

### `Derived_Feature_Rules`

| カラム | 内容 |
|---|---|
| `Rule_ID` | 重複しないルール名 |
| `Scope_Type` | `group` / `factor` |
| `Scope_Value` | 対象 |
| `Feature_Type` | `difference` / `rolling_mean_deviation` / `rolling_mean_ratio` / `expanding_mean_deviation` |
| `Difference_Periods` | 差分期間 |
| `Window_Periods` | 移動平均窓 |
| `Min_Periods` | 最低履歴数 |
| `Source_Lag_Periods` | スコア時点から元FAを何期遅らせるか |
| `Exclude_Source_From_Baseline` | 最新値を移動平均から除くか |
| `Enabled` | ルール有効可否 |
| `Selected` | selectedモードで使用するか |
| `Direction_Mode` | `inherit` / `reverse` / `custom` |
| `Custom_Direction` | custom時の1/-1 |

## 8. 入力チェックリスト

```text
□ date + ISINが一意
□ stock_returnの時点定義を確認
□ FA列名とFactor_Codeが一致
□ Directionが1または-1
□ market_capが正
□ countryとsectorの表記揺れがない
□ 欠損を0埋めしていない
□ 派生特徴量のSource_Lag_Periodsを確認
```
