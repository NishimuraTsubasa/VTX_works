# LightGBMテクニカル指標モデル 診断・改善分析 指示書

## 1. 分析目的

テクニカル指標を特徴量としてLightGBMを用いて株式の将来リターンを予測しているが、現在以下の問題が確認されている。

- 目的変数：20営業日先リターン
- モデル：LightGBM
- 特徴量：テクニカル指標中心
- 五分位分析では、予測リターンが低い銘柄ほど実現リターンが高いなど、想定と逆方向の結果が確認された
- 一方、LightGBM予測値のRank ICは時点によって正負が大きく変化しており、安定していない
- 現状では予測値を単純に反転して利用できるとは判断できない

本分析では、以下を順番に検証し、モデル不調の原因を特定する。

1. 目的変数の定義に問題がないか
2. 各特徴量単体に将来リターンの予測力が存在するか
3. 負のICが安定した逆張りシグナルなのか、単なるノイズなのか
4. 各特徴量と20営業日という予測ホライズンが適合しているか
5. LightGBMが特徴量の情報を改善しているか、逆に破壊しているか
6. LightGBMが過学習していないか
7. 特定の市場レジームでのみモデルが有効なのか
8. 最終的に投資戦略として利用可能なモデルなのか

---

# 2. データ前提

最低限、以下のカラムを含むパネルデータを使用する。

| カラム | 内容 |
|---|---|
| `date` | 基準日 |
| `ticker` | 銘柄コード |
| `price` | 基準日時点の株価 |
| `future_return_20d` | 20営業日先リターン |
| 各種特徴量 | RSI、モメンタム、移動平均乖離、ボラティリティ等 |
| `prediction` | LightGBMによる予測値 |

データ構造は原則として、

**1行 = 1日 × 1銘柄**

とする。

---

# 3. Step 1：目的変数の定義確認

最初に20営業日先リターンの作成方法が正しいことを確認する。

目的変数は原則として以下とする。

$$
R_{i,t}^{20D}
=
\frac{P_{i,t+20}}{P_{i,t}}-1
$$

Pythonでは概念的に以下となる。

```python
df = df.sort_values(["ticker", "date"])

df["future_return_20d"] = (
    df.groupby("ticker")["price"].shift(-20)
    / df["price"]
    - 1
)
```

以下を重点的に確認すること。

- `shift(-20)` になっているか
- 誤って `shift(+20)` になっていないか
- 分子と分母が逆転していないか
- 銘柄ごとにshiftされているか
- 日付順に正しくsortされているか
- 株式分割等を調整した価格を使用しているか
- 必要に応じて配当込みリターンを使用しているか
- 上場廃止銘柄などが不自然に除外されていないか
- 将来情報が特徴量に混入していないか
- 当日の終値を利用できないタイミングで当日終値由来の特徴量を使用していないか

目的変数に問題が見つかった場合は、以降の分析より先に修正すること。

---

# 4. Step 2：特徴量単体のCross-sectional Rank IC分析

## 4.1 Rank ICの定義

各時点について、複数銘柄を横断して特徴量と将来リターンのSpearman順位相関を計算する。

特徴量 $j$ の時点 $t$ におけるRank ICを以下とする。

$$
IC_{j,t}
=
\operatorname{SpearmanCorr}_{i}
\left(
X_{j,i,t},
R_{i,t}^{20D}
\right)
$$

ここで重要なのは、相関を取る方向が**時間方向ではなく銘柄方向**である点である。

例えば2025年1月6日に500銘柄存在する場合、

- 500銘柄のRSI
- 同じ500銘柄の20営業日先リターン

についてSpearman順位相関を計算する。

これによって2025年1月6日のRSI Rank ICが1つ得られる。

これを各営業日について繰り返し、ICの時系列を作成する。

---

## 4.2 特徴量ごとのRank IC計算

例：

```python
import pandas as pd
import numpy as np

features = [
    "RSI",
    "momentum_5d",
    "momentum_20d",
    "ma_deviation",
    "volatility_20d",
]

def calc_rank_ic(group, feature, target):
    tmp = group[[feature, target]].dropna()

    if len(tmp) < 10:
        return np.nan

    if tmp[feature].nunique() < 2:
        return np.nan

    if tmp[target].nunique() < 2:
        return np.nan

    return tmp[feature].corr(
        tmp[target],
        method="spearman"
    )

ic_dict = {}

for feature in features:
    ic_dict[feature] = (
        df.groupby("date")
        .apply(
            lambda x: calc_rank_ic(
                x,
                feature,
                "future_return_20d"
            )
        )
    )

ic_df = pd.DataFrame(ic_dict)
```

最終的に以下のようなデータを作成する。

| Date | RSI | Mom5d | Mom20d | MA乖離 | Volatility |
|---|---:|---:|---:|---:|---:|
| Day 1 | -0.05 | -0.08 | +0.03 | -0.04 | +0.01 |
| Day 2 | -0.02 | -0.06 | +0.05 | -0.01 | +0.04 |
| Day 3 | +0.01 | -0.03 | +0.02 | -0.02 | +0.03 |

---

# 5. Step 3：特徴量ごとのICサマリー

各特徴量について最低限以下を算出する。

- Mean IC
- Median IC
- IC Standard Deviation
- ICIR
- IC > 0 の割合
- IC < 0 の割合
- 観測数

ICIRは以下とする。

$$
ICIR
=
\frac{\operatorname{Mean}(IC_t)}
{\operatorname{Std}(IC_t)}
$$

Python例：

```python
ic_summary = pd.DataFrame({
    "mean_ic": ic_df.mean(),
    "median_ic": ic_df.median(),
    "std_ic": ic_df.std(),
    "icir": ic_df.mean() / ic_df.std(),
    "positive_ratio": (ic_df > 0).mean(),
    "negative_ratio": (ic_df < 0).mean(),
    "n_obs": ic_df.count(),
})

ic_summary = ic_summary.sort_values(
    "mean_ic",
    ascending=False
)

print(ic_summary)
```

出力イメージ：

| Feature | Mean IC | Std IC | ICIR | Positive Ratio |
|---|---:|---:|---:|---:|
| RSI | -0.031 | 0.080 | -0.39 | 34% |
| Mom5d | -0.025 | 0.090 | -0.28 | 39% |
| Mom20d | +0.018 | 0.070 | +0.26 | 61% |
| Volatility | +0.003 | 0.060 | +0.05 | 51% |

---

# 6. Step 4：負のICの解釈

Mean ICが負であることだけを理由に、その特徴量を無効と判断してはいけない。

例えば、

$$
Mean\ IC=-0.03
$$

であり、複数期間にわたって安定して負であれば、

> 特徴量が高い銘柄ほど、その後20営業日のリターンが低い

という逆張り方向の有効なシグナルである可能性がある。

以下の3パターンに分類すること。

## パターンA：安定して正

順張り方向のシグナルとして利用候補とする。

## パターンB：安定して負

逆張り方向のシグナルとして利用候補とする。

解釈上必要であれば、

$$
X^*=-X
$$

として向きを統一してもよい。

ただしLightGBMは特徴量と目的変数の非線形関係を学習できるため、LightGBM投入前に必ず符号を反転する必要はない。

## パターンC：正負が頻繁に反転

単体特徴量として安定した予測力がない、または市場レジーム依存である可能性を疑う。

このケースでは単純な符号反転を行わない。

---

# 7. Step 5：Rolling IC分析

Mean ICだけではシグナルの時間的な安定性を判断できないため、Rolling ICを確認する。

まず60営業日を基本とする。

```python
rolling_ic_60 = ic_df.rolling(60).mean()

rolling_ic_120 = ic_df.rolling(120).mean()
```

各特徴量について、

- Daily Rank IC
- 60営業日Rolling Mean IC
- 必要に応じて120営業日Rolling Mean IC

をグラフ化する。

例：

```python
import matplotlib.pyplot as plt

for feature in features:
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        ic_df.index,
        ic_df[feature],
        alpha=0.25,
        label="Daily IC"
    )

    ax.plot(
        rolling_ic_60.index,
        rolling_ic_60[feature],
        label="60D Rolling IC"
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(f"{feature}: Rank IC")
    ax.legend()

    plt.show()
```

確認事項：

- ICの符号が長期間維持されているか
- 特定期間だけ有効ではないか
- 最近シグナルが消失していないか
- 正負の反転が頻繁に起きていないか

---

# 8. Step 6：年別IC分析

各特徴量のRank ICを年別に集計する。

```python
ic_df.index = pd.to_datetime(ic_df.index)

yearly_ic = ic_df.groupby(
    ic_df.index.year
).mean()

print(yearly_ic)
```

例：

| Feature | 2022 | 2023 | 2024 | 2025 |
|---|---:|---:|---:|---:|
| RSI | -0.04 | -0.03 | -0.05 | -0.02 |
| Mom5d | -0.03 | +0.02 | -0.04 | +0.01 |
| Mom20d | +0.02 | +0.03 | +0.01 | +0.02 |

この場合、

- RSI：安定して負
- Mom20d：比較的安定して正
- Mom5d：符号が不安定

と判断する。

単純な全期間平均だけでなく、**年ごとの符号の一貫性を重視すること。**

---

# 9. Step 7：予測ホライズン別IC分析

現在の目的変数は20営業日先リターンである。

しかし、テクニカル特徴量の有効期間と20営業日が一致している保証はない。

以下の予測ホライズンについて将来リターンを作成する。

- 1営業日
- 5営業日
- 10営業日
- 20営業日
- 40営業日

$$
R_{i,t}^{h}
=
\frac{P_{i,t+h}}{P_{i,t}}-1
$$

Python例：

```python
horizons = [1, 5, 10, 20, 40]

df = df.sort_values(["ticker", "date"])

for h in horizons:
    df[f"future_return_{h}d"] = (
        df.groupby("ticker")["price"].shift(-h)
        / df["price"]
        - 1
    )
```

各特徴量について各ホライズンの平均Rank ICを計算する。

```python
horizon_ic = pd.DataFrame(
    index=features,
    columns=horizons,
    dtype=float
)

for feature in features:

    for h in horizons:

        target = f"future_return_{h}d"

        temp_ic = (
            df.groupby("date")
            .apply(
                lambda x: calc_rank_ic(
                    x,
                    feature,
                    target
                )
            )
        )

        horizon_ic.loc[feature, h] = temp_ic.mean()

print(horizon_ic)
```

出力イメージ：

| Feature | 1D | 5D | 10D | 20D | 40D |
|---|---:|---:|---:|---:|---:|
| RSI | -0.04 | -0.06 | -0.05 | -0.01 | 0.00 |
| Mom5d | -0.03 | -0.04 | -0.02 | 0.00 | +0.01 |
| Mom20d | 0.00 | +0.01 | +0.02 | +0.03 | +0.02 |

この分析によって、

> 特徴量そのものに予測力がない

のか、

> 特徴量には予測力があるが20営業日という予測期間と合っていない

のかを切り分ける。

---

# 10. Step 8：LightGBM予測値のRank IC分析

LightGBMの予測値についても、各時点でクロスセクショナルRank ICを計算する。

$$
IC_t^{Model}
=
\operatorname{SpearmanCorr}_i
\left(
\hat{R}_{i,t},
R_{i,t}^{20D}
\right)
$$

Python例：

```python
model_ic = (
    df.groupby("date")
    .apply(
        lambda x: calc_rank_ic(
            x,
            "prediction",
            "future_return_20d"
        )
    )
)

model_ic_summary = {
    "mean_ic": model_ic.mean(),
    "median_ic": model_ic.median(),
    "std_ic": model_ic.std(),
    "icir": model_ic.mean() / model_ic.std(),
    "positive_ratio": (model_ic > 0).mean(),
    "negative_ratio": (model_ic < 0).mean(),
    "n_obs": model_ic.count(),
}

print(model_ic_summary)
```

以下を確認する。

- Mean Rank IC
- Median Rank IC
- Standard Deviation
- ICIR
- Positive Ratio
- Negative Ratio
- 年別Rank IC
- Rolling Rank IC

---

# 11. Step 9：特徴量単体とLightGBMを比較

各特徴量とLightGBMのOOS Rank ICを比較する。

例：

| Signal | Mean IC | ICIR | Positive Ratio |
|---|---:|---:|---:|
| RSI | -0.025 | -0.31 | 37% |
| Mom5d | -0.018 | -0.20 | 41% |
| Mom20d | +0.021 | +0.30 | 62% |
| Volatility | -0.010 | -0.15 | 44% |
| LightGBM | +0.001 | +0.01 | 50% |

この結果の場合、

> 個別特徴量には一定の情報が存在するにもかかわらず、LightGBMで統合した結果、その情報が消えている

可能性を疑う。

逆に、

- 個別特徴量 IC ≒ 0
- LightGBM IC ≒ 0

であれば、

> そもそも入力特徴量に20営業日先リターンを予測する情報がほぼ存在していない

可能性を優先する。

---

# 12. Step 10：Train / Validation / TestのRank IC比較

LightGBMについて、

- Train
- Validation
- Test

それぞれでRank ICを計算する。

ランダムシャッフルによるTrain/Test分割は使用しない。

時系列順に分割すること。

例：

```text
Train      : 2015-2021
Validation : 2022-2023
Test       : 2024-2025
```

各期間について以下を比較する。

| Dataset | Mean Rank IC | ICIR | Positive Ratio |
|---|---:|---:|---:|
| Train | | | |
| Validation | | | |
| Test | | | |

例えば、

| Dataset | Mean Rank IC |
|---|---:|
| Train | +0.15 |
| Validation | +0.01 |
| Test | -0.02 |

であれば、典型的な過学習を疑う。

---

# 13. Step 11：LightGBMの過学習診断

過学習が疑われる場合は以下を確認する。

## 主なLightGBMパラメータ

- `num_leaves`
- `max_depth`
- `min_data_in_leaf`
- `min_gain_to_split`
- `feature_fraction`
- `bagging_fraction`
- `bagging_freq`
- `lambda_l1`
- `lambda_l2`
- `learning_rate`
- `n_estimators`

モデルをさらに複雑にするのではなく、まず正則化を強化する。

基本的な方向性：

```text
num_leaves          ↓
max_depth           ↓
min_data_in_leaf    ↑
feature_fraction    ↓
bagging_fraction    ↓
lambda_l1           ↑
lambda_l2           ↑
```

評価指標はRMSEだけに依存せず、

**Out-of-Sample Rank IC**

を主要評価指標の一つとする。

---

# 14. Step 12：単純モデルとの比較

LightGBMを使用する価値が本当に存在するか確認する。

最低限以下をベンチマークモデルとする。

1. 単一特徴量モデル
2. 特徴量単純平均
3. IC加重スコア
4. Linear Regression
5. Ridge Regression
6. LightGBM

特徴量を日次クロスセクションでRank化してから利用する方法も検討する。

```python
rank_features = []

for feature in features:

    rank_col = f"{feature}_rank"

    df[rank_col] = (
        df.groupby("date")[feature]
        .rank(
            pct=True,
            method="average"
        )
    )

    rank_features.append(rank_col)
```

単純平均スコア：

```python
df["equal_weight_score"] = (
    df[rank_features].mean(axis=1)
)
```

IC加重スコアの概念：

$$
Score_{i,t}
=
\sum_j w_j X_{j,i,t}
$$

ここで $w_j$ は学習期間内で計算した特徴量IC等を利用する。

**Test期間の情報を使って重みを作成してはいけない。**

LightGBMがOut-of-Sampleでこれら単純モデルを上回らない場合、

> 複雑な非線形モデルを使用するメリットが十分確認できない

と判断する。

---

# 15. Step 13：特徴量間相関の確認

テクニカル指標には情報の重複が多い可能性がある。

例えば以下は同じ価格系列から派生する。

- RSI
- Stochastic
- 短期リターン
- 中期リターン
- 移動平均乖離率
- MACD
- Bollinger Band関連
- 過去ボラティリティ

特徴量間のSpearman相関を確認する。

ただし、全期間を単純に縦に結合した相関だけでなく、可能であれば**日次クロスセクショナル相関の平均**も確認する。

単純な確認：

```python
feature_corr = df[features].corr(
    method="spearman"
)

print(feature_corr)
```

可能であれば以下も実施する。

- 相関クラスタリング
- 高相関特徴量の削減
- 特徴量カテゴリ別モデル
- Momentum系
- Reversal系
- Volatility系
- Trend系

などへの分類。

---

# 16. Step 14：市場レジーム別Rank IC分析

全期間でRank ICが0付近でも、特定の市場環境では有効な可能性がある。

以下のようなレジームで分割して分析する。

## 市場方向

- Bull Market
- Bear Market

## ボラティリティ

- High Volatility
- Low Volatility

## 銘柄属性

- Large Cap
- Small Cap
- High Beta
- Low Beta

必要に応じて、

- Value / Growth
- Sector
- Liquidity
- 金利環境
- VIX等の市場ボラティリティ
- 市場トレンド

も分析する。

出力例：

| Regime | LightGBM Mean IC | ICIR |
|---|---:|---:|
| Bull Market | +0.01 | +0.10 |
| Bear Market | -0.06 | -0.60 |
| High Vol | -0.08 | -0.75 |
| Low Vol | +0.03 | +0.35 |

このような結果なら、

> モデルが完全に無効なのではなく、市場環境によって予測方向または予測力が変化している

可能性を検討する。

---

# 17. Step 15：SHAPによるLightGBM挙動確認

LightGBMについてSHAP分析を実施する。

確認事項：

- どの特徴量が予測値を支配しているか
- RSI上昇時に予測値が上がるのか下がるのか
- モメンタム上昇時に予測値がどう変化するか
- 極端値で不自然な予測をしていないか
- 一部特徴量に過度に依存していないか
- 単体ICでは有効な特徴量をLightGBMが逆方向に使用していないか

可能であれば時期別にSHAPを計算して、

> 特徴量と予測値の関係自体がレジームによって変化していないか

も確認する。

---

# 18. Step 16：20営業日ターゲットのオーバーラップへの対応

20営業日先リターンを毎営業日作成すると、隣接する目的変数が大きく重複する。

例えば、

```text
1月6日 → 1月6日から約20営業日後
1月7日 → 1月7日から約20営業日後
```

となり、両者は19営業日前後の期間を共有する。

そのため、

- Daily IC
- 将来リターン
- バックテストリターン

には強い自己相関が生じ得る。

通常の独立同分布を仮定したt検定をそのまま使用すると、統計的有意性を過大評価する可能性がある。

以下の対応を検討する。

## 方法1：Non-overlapping Sample

20営業日ごとにデータを抽出して評価する。

## 方法2：Purged Time Series Cross Validation

学習期間と検証期間の境界から、ターゲット期間が重複するサンプルを除外する。

## 方法3：Embargo

Validation/Test期間の直前について一定期間を学習対象から除外する。

**ランダムK-Fold Cross Validationは原則使用しない。**

---

# 19. Step 17：五分位分析

最終的な投資シグナルとして評価するため、各時点でLightGBM予測値を五分位に分割する。

- Q1：予測値最低
- Q2
- Q3
- Q4
- Q5：予測値最高

例：

```python
def assign_quintile(group):

    group = group.copy()

    group["quintile"] = pd.qcut(
        group["prediction"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    )

    return group

df = (
    df.groupby("date", group_keys=False)
    .apply(assign_quintile)
)
```

各Quintileの実現リターンを算出する。

```python
quintile_return = (
    df.groupby(["date", "quintile"])[
        "future_return_20d"
    ]
    .mean()
    .unstack()
)
```

以下を確認する。

1. Q1からQ5に単調性があるか
2. Q5 - Q1
3. Q1 - Q5
4. 年別Long-Short Spread
5. 勝率
6. Turnover
7. Transaction Cost
8. Long側とShort側のどちらが収益源なのか

例：

| Quintile | Return |
|---|---:|
| Q1 | +2.0% |
| Q2 | +1.3% |
| Q3 | +0.5% |
| Q4 | -0.2% |
| Q5 | -1.0% |

この結果だけを見て、

```python
signal = -prediction
```

と変更してはいけない。

まずOOS Rank ICの符号と五分位構造が複数期間にわたって安定していることを確認する。

---

# 20. Step 18：ランダムモデルとの比較

現在のLightGBMがランダム予測より本当に優れているか検証する。

各時点で銘柄にランダムスコアを付与し、

- Mean Rank IC
- ICIR
- 五分位スプレッド

を計算する。

これを多数回繰り返し、Null Distributionを作成する。

例：

```python
n_simulations = 1000

random_mean_ics = []

for seed in range(n_simulations):

    rng = np.random.default_rng(seed)

    temp = df[
        ["date", "ticker", "future_return_20d"]
    ].copy()

    temp["random_score"] = rng.normal(
        size=len(temp)
    )

    random_ic = (
        temp.groupby("date")
        .apply(
            lambda x: calc_rank_ic(
                x,
                "random_score",
                "future_return_20d"
            )
        )
    )

    random_mean_ics.append(
        random_ic.mean()
    )
```

LightGBMのMean Rank ICが、ランダムモデルによるIC分布のどの位置に存在するか確認する。

例えば、

```python
random_mean_ics = np.asarray(
    random_mean_ics
)

model_mean_ic = model_ic.mean()

percentile = (
    random_mean_ics < model_mean_ic
).mean()

print(percentile)
```

LightGBMの結果がランダムモデルの一般的な範囲内であれば、

> 現時点ではモデルに統計的・経済的に明確な予測力を確認できない

と判断する。

---

# 21. Step 19：必要に応じてRankingモデルを検討

最終目的が絶対リターンの正確な予測ではなく、

> 将来リターンが相対的に高い銘柄を上位に順位付けすること

であれば、RegressionだけでなくRanking Objectiveの利用を検討する。

例えばLightGBM Ranker等を候補とする。

ただし、

> RegressionのRank ICが悪いからRankerを使えば解決する

とは考えないこと。

まず特徴量自体にクロスセクショナルな予測力が存在することをStep 2〜7で確認した後に検討する。

---

# 22. 最終診断ルール

## ケースA：個別特徴量のRank ICもほぼ0

### 判断

入力特徴量に十分な予測情報が存在しない可能性が高い。

### 対応

以下を優先する。

- 特徴量の再設計
- 新規特徴量の追加
- ファンダメンタル特徴量
- センチメント特徴量
- クロスセクショナル特徴量
- 市場レジーム特徴量
- 予測ホライズン変更

LightGBMのハイパーパラメータ調整を最優先にはしない。

---

## ケースB：個別特徴量にはRank ICがあるがLightGBMでは消える

### 判断

LightGBMが特徴量に含まれる情報をうまく統合できていない可能性がある。

### 対応

- LightGBMの単純化
- 正則化強化
- 特徴量削減
- Ridgeとの比較
- 単純IC加重モデルとの比較
- Ranking Objectiveの検討

---

## ケースC：Trainでは良いがValidation/TestでICが消える

### 判断

過学習の可能性が高い。

### 対応

- モデル複雑度低下
- 正則化強化
- 特徴量削減
- Purged CV
- Embargo
- Walk-forward validation
- Hyperparameter探索範囲の縮小

---

## ケースD：特定レジームのみRank ICが存在

### 判断

条件付きシグナルである可能性がある。

### 対応

- Market Regimeを特徴量として追加
- Regime別モデル
- Regime Gating
- 有効局面のみポジションを取る方法の検討

ただし、レジーム分類自体に将来情報を利用しないこと。

---

## ケースE：OOSでも安定してRank ICが負

### 判断

予測方向が逆である可能性がある。

### 対応

OOSで継続的に負のICが確認される場合に限り、

$$
AlphaScore=-Prediction
$$

の利用を検討する。

ただし、バックテスト結果を確認した後に符号を反転することはデータマイニングとなり得るため、別の完全なHoldout期間で再検証する。

---

## ケースF：Rank ICの符号が時点ごとにランダム

### 判断

現時点ではモデルが将来のクロスセクショナル順位を安定して予測できていない可能性が高い。

### 対応

予測値の単純反転は行わない。

以下を優先する。

1. 特徴量単体ICの確認
2. ホライズン別IC
3. レジーム別IC
4. Train/OOS比較
5. 単純モデルとの比較
6. 特徴量設計の見直し

---

# 23. 分析実施の優先順位

以下の順番で分析を実施すること。

```text
1. 目的変数の定義確認
        ↓
2. 各特徴量のCross-sectional Rank IC
        ↓
3. Mean IC / ICIR / Positive Ratio
        ↓
4. Rolling IC
        ↓
5. 年別IC
        ↓
6. 1D / 5D / 10D / 20D / 40D ホライズン別IC
        ↓
7. LightGBM予測値のRank IC
        ↓
8. 単体特徴量 vs LightGBM
        ↓
9. Train / Validation / Test IC
        ↓
10. Ridge・単純モデルとの比較
        ↓
11. 特徴量間相関
        ↓
12. 市場レジーム分析
        ↓
13. SHAP分析
        ↓
14. 五分位分析
        ↓
15. ランダムモデルとの比較
        ↓
16. 取引コスト込みバックテスト
```

---

# 24. 最終成果物

分析終了時には最低限以下を出力すること。

## ① 特徴量ICサマリー

| Feature | Mean IC | Median IC | Std IC | ICIR | Positive % | Negative % |
|---|---:|---:|---:|---:|---:|---:|

---

## ② Rolling ICグラフ

各主要特徴量について、

- Daily Rank IC
- 60営業日Rolling Rank IC

を表示する。

---

## ③ 年別Rank IC

| Feature | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|

---

## ④ ホライズン別Rank IC

| Feature | 1D | 5D | 10D | 20D | 40D |
|---|---:|---:|---:|---:|---:|

---

## ⑤ モデル比較

| Model | Train IC | Validation IC | Test IC | Test ICIR |
|---|---:|---:|---:|---:|
| Single Factor | | | | |
| Equal Weight | | | | |
| IC Weighted | | | | |
| Ridge | | | | |
| LightGBM | | | | |

---

## ⑥ 市場レジーム別IC

| Regime | Mean IC | ICIR | Positive Ratio |
|---|---:|---:|---:|

---

## ⑦ 五分位分析

| Quintile | Gross Return | Net Return |
|---|---:|---:|
| Q1 | | |
| Q2 | | |
| Q3 | | |
| Q4 | | |
| Q5 | | |

加えて以下を算出する。

- Q5 - Q1
- Q1 - Q5
- Turnover
- Transaction Cost
- Long側リターン
- Short側リターン
- 年別Long-Short Return
- Long-Short勝率

---

# 25. 最終分析レポートで回答すべき事項

最終的な分析結果では、最低限以下について明確な結論を記載すること。

1. テクニカル特徴量単体に20営業日先リターンの予測力があるか
2. どの特徴量に最も強いRank ICが存在するか
3. 負のICは安定した逆張り効果なのか、単なるノイズなのか
4. ICの符号は時間を通じて安定しているか
5. 20営業日という予測ホライズンは特徴量に適しているか
6. より適切な予測ホライズンが存在するか
7. LightGBMは単純モデルよりOOS Rank ICを改善しているか
8. LightGBMに過学習が発生しているか
9. 特定市場レジームへの依存性があるか
10. 五分位リターンの逆転は再現性のある現象か
11. 予測値の符号を反転して利用する合理性があるか
12. 現モデルを「継続利用」「修正」「廃棄」のどれと判断するか
13. 次に最優先で改善すべき項目は何か

---

# 26. 分析上の最重要ルール

今回最も注意すべき点は、

> 五分位リターンが一度逆方向になったという事実だけを理由として、LightGBMの予測値を反転しないこと

である。

特に現在は、

> LightGBM予測値のRank ICが時点によって正負に大きく変化している

ため、

```python
alpha_score = -prediction
```

とするだけでは問題の根本解決にならない。

まず、

**特徴量 → 目的変数**

の段階で予測情報が存在するか確認する。

その後、

**特徴量 → LightGBM → Prediction**

の過程で情報が改善されているのか、失われているのかを検証する。

モデル評価の中心は、

**Out-of-Sample Cross-sectional Rank IC**

とする。

最終的には、

**Rank ICの安定性 + 五分位単調性 + Long-Short収益 + 取引コスト + 時系列安定性**

を総合してモデルの有効性を判断すること。