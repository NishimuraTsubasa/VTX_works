# データ辞書

## 1. factor_master.xlsx

### Factor_Master

| 列 | 必須 | 内容 |
|---|---:|---|
| Factor_Code | Yes | ファクターデータの物理列名。例：FA0101 |
| Factor_Name_JP | Yes | 日本語表示名 |
| Factor_Name_EN | 任意 | 英語表示名 |
| Factor_Group | Yes | Value、Momentum、Qualityなど |
| Enabled | Yes | 1：使用、0：停止 |
| Direction | Yes | 1：高いほど望ましい、-1：低いほど望ましい |
| Base_Weight | Yes | manual統合時の基準ウェイト |
| Transform | Yes | default / none / log / log1p / inverse / signed_log |
| Winsorize | Yes | default / none / 1_99 / 2.5_97.5 / mad_3 |
| Neutralize | Yes | default / 1 / 0 |
| Rank_Normalize | Yes | default / 1 / 0 |
| Min_Coverage | Yes | 0～1の最低観測率 |
| Description | 任意 | 定義・データ取得上の注意 |

### Group_Settings

| 列 | 内容 |
|---|---|
| Factor_Group | Factor_Masterと対応するグループ名 |
| Display_Name | 出力上の表示名 |
| Enabled | 1：使用、0：停止 |
| Aggregation_Method | equal_weight / manual / ic_adjusted / pca |
| Lookback_Periods | IC・PCA推定期間 |
| Min_Periods | 推定開始に必要な最低期間 |
| Max_Weight | 1ファクターのウェイト上限 |
| Weight_Smoothing | 前期ウェイトを残す比率 |
| Fallback_Method | 推定不能時の方法 |

### Group_Method_Params

| 列 | 内容 |
|---|---|
| Factor_Group | 対象グループ |
| Param_Name | 上書きするパラメータ名 |
| Param_Value | パラメータ値 |

## 2. factors_and_returns.xlsx

| 列 | 必須 | 内容 |
|---|---:|---|
| date | Yes | ファクター観測日 |
| ISIN | Yes | 個別銘柄キー |
| stock_return | Yes | 観測日から翌期までの個別銘柄リターン |
| sector | Yes | セクター |
| country | Yes | 国・地域。ConfigでJPY、USD等の選定キーへ変換可能 |
| market_cap | 推奨 | 時価総額。中立化・代表銘柄選定に使用 |
| FAxxxx | Yes | Factor_Masterで有効化したファクター列 |

## 3. index_constituents.xlsx

1シート1指数。シート名を指数名として使用します。

| 列 | 必須 | 内容 |
|---|---:|---|
| ISIN | Yes | 実際の指数構成銘柄識別子 |
| sector | Yes | セクター |
| country | 任意 | 国・地域 |

## 4. index_sector_weights.xlsx

行がセクター、列が指数名です。値は0～1または0～100です。

## 5. futures_returns.xlsx

行が日付、列が指数名です。`monthly_returns`または`weekly_returns`をConfigで選択します。
